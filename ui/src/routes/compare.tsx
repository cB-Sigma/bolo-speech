import { Cloud, Columns3, FileUp, Loader2, Play, Square, TriangleAlert, Zap } from 'lucide-react'
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { CloudMark } from '@/components/model-bits'
import { Empty, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { describeError, useCatalog, useSamples } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { CatalogModel } from '@/lib/api/types'
import { CLOUD_NOTICE, isCloud } from '@/lib/capabilities'
import {
  comparable,
  compareSpeech,
  compareTranscription,
  fastest,
  MAX_COMPARE,
  MIN_COMPARE,
  type CompareResult,
} from '@/lib/compare'
import { describePrice, fmtCost } from '@/lib/cost'
import { fmtMs, fmtRtf, fmtSeconds } from '@/lib/format'
import { cn } from '@/lib/utils'

type Task = 'stt' | 'tts'
const UPLOAD = '__upload'
const DEFAULT_TEXT = 'Your order of 3 items will arrive on 12 March. Thank you for calling.'

function ModelChoice({
  models,
  value,
  onChange,
}: {
  models: CatalogModel[]
  value: string[]
  onChange: (ids: string[]) => void
}) {
  const id = useId()
  const full = value.length >= MAX_COMPARE
  const groups = [
    { label: 'Runs on this server', items: models.filter((m) => !isCloud(m)) },
    { label: 'Cloud', items: models.filter((m) => isCloud(m)) },
  ].filter((g) => g.items.length > 0)
  if (models.length === 0) {
    return <p className="text-[13px] text-muted-foreground">No model of this kind is installed.</p>
  }
  return (
    <div className="grid gap-3">
      {groups.map((g) => (
        <fieldset key={g.label} className="min-w-0">
          <legend className="mb-1.5 eyebrow">{g.label}</legend>
          <ul className="grid max-h-52 gap-x-4 gap-y-1.5 overflow-y-auto sm:grid-cols-2">
            {g.items.map((m) => {
              const on = value.includes(m.id)
              return (
                <li key={m.id} className="flex min-w-0 items-center gap-2">
                  <Checkbox
                    id={`${id}-${m.id}`}
                    checked={on}
                    disabled={!on && full}
                    onCheckedChange={(v) => onChange(v === true ? [...value, m.id] : value.filter((x) => x !== m.id))}
                  />
                  <Label htmlFor={`${id}-${m.id}`} className="min-w-0 truncate num text-xs font-normal">
                    {m.id}
                  </Label>
                </li>
              )
            })}
          </ul>
        </fieldset>
      ))}
    </div>
  )
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="contents">
      <dt className="text-muted-foreground" title={hint}>
        {label}
      </dt>
      <dd className="text-right num">{value}</dd>
    </div>
  )
}

function ResultCard({
  result,
  model,
  task,
  best,
}: {
  result: CompareResult
  model: CatalogModel | undefined
  task: Task
  best: boolean
}) {
  const cloud = isCloud(model)
  const price = describePrice(model?.provider?.price)
  return (
    <article
      className={cn('flex min-w-0 flex-col rounded-xl border bg-card', best ? 'border-good/60' : 'border-border')}
      data-testid={`compare-result-${result.model}`}
      aria-busy={result.status === 'running'}
    >
      <header className="flex min-h-11 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <h3 className="min-w-0 flex-1 truncate num text-[13px] font-semibold">{result.model}</h3>
        {model && <CloudMark model={model} />}
        {best && (
          <span className="inline-flex items-center gap-1 text-xs text-good-text">
            <Zap className="size-3.5" aria-hidden /> Fastest
          </span>
        )}
      </header>
      <div className="grid flex-1 content-start gap-3 p-4">
        {result.status === 'running' ? (
          <div className="grid gap-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        ) : result.status === 'error' ? (
          <p className="flex items-start gap-1.5 text-[13px] leading-5 text-critical-text" role="alert">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            <span>
              {result.error}
              {result.errorCode ? <span className="num text-muted-foreground"> ({result.errorCode})</span> : null}
            </span>
          </p>
        ) : task === 'stt' ? (
          <p className="text-[14px] leading-6" data-testid="compare-text">
            {result.text || <span className="text-muted-foreground">(no speech recognised)</span>}
          </p>
        ) : (
          <audio controls src={result.audioUrl} className="h-9 w-full" aria-label={`Speech from ${result.model}`} />
        )}
        {result.answeredBy && (
          <p className="text-xs leading-5 text-warning-text" data-testid="compare-fallback">
            Answered by <span className="num">{result.answeredBy}</span>: the model asked for could not answer
            {result.fallbackReason ? ` (${result.fallbackReason})` : ''}.
          </p>
        )}
        {result.status !== 'running' && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            <Metric
              label="Server time"
              value={fmtMs(result.serverMs)}
              hint="From the request's trace: receiving, queueing and processing on the server"
            />
            {task === 'tts' && (
              <Metric
                label="First audio"
                value={fmtMs(result.ttfaMs ?? result.firstByteMs)}
                hint="Until the first audio left the server (or, without a trace, reached the browser)"
              />
            )}
            <Metric label="In the browser" value={fmtMs(result.clientMs)} hint="Request sent to last byte received" />
            {cloud && (
              <Metric
                label="At the provider"
                value={fmtMs(result.providerMs)}
                hint="Time spent waiting for the cloud API"
              />
            )}
            <Metric label="Real-time factor" value={fmtRtf(result.rtf)} hint="Processing time ÷ audio duration" />
            {task === 'tts' && <Metric label="Audio" value={fmtSeconds(result.audioSeconds)} />}
            <Metric
              label="Estimated cost"
              value={result.cost ? fmtCost(result.cost) : cloud ? (price ? '–' : 'no price set') : 'none (runs here)'}
              hint={
                cloud
                  ? (price ?? 'Add what you pay on the Providers screen to see an estimate. It is never a bill.')
                  : 'A local model has no per-request cost'
              }
            />
          </dl>
        )}
      </div>
    </article>
  )
}

/** The same audio or the same text through two or more models, side by side: output, latency, estimated cost. */
export default function ComparePage() {
  const catalog = useCatalog()
  const samples = useSamples()
  const id = useId()
  const [task, setTask] = useState<Task>('stt')
  const [chosen, setChosen] = useState<Record<Task, string[] | null>>({ stt: null, tts: null })
  const [sample, setSample] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState('')
  const [text, setText] = useState(DEFAULT_TEXT)
  const [results, setResults] = useState<CompareResult[]>([])
  const [ranTask, setRanTask] = useState<Task>('stt')
  const [running, setRunning] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const urls = useRef<string[]>([])

  const all = useMemo(() => catalog.data?.models ?? [], [catalog.data])
  const byId = useMemo(() => new Map(all.map((m) => [m.id, m])), [all])
  const models = useMemo(() => comparable(all, task), [all, task])
  // Default: the first local and the first cloud model, so the comparison the page is for shows up by itself.
  const preset = useMemo(() => {
    const local = models.filter((m) => !isCloud(m))
    const cloud = models.filter((m) => isCloud(m))
    return [...local.slice(0, cloud.length ? 1 : 2), ...cloud.slice(0, 1)].map((m) => m.id)
  }, [models])
  const picked = (chosen[task] ?? preset).filter((x) => byId.has(x))
  const cloudPicked = picked.filter((x) => isCloud(byId.get(x)))
  const audioSamples = (samples.data?.samples ?? []).filter((s) => s.kind === 'audio' && s.url)
  const source = sample ?? (file ? UPLOAD : (audioSamples[0]?.id ?? UPLOAD))

  const revoke = () => {
    for (const u of urls.current) URL.revokeObjectURL(u)
    urls.current = []
  }
  useEffect(
    () => () => {
      abort.current?.abort()
      revoke()
    },
    [],
  )

  const stop = () => {
    abort.current?.abort()
    setRunning(false)
  }

  const run = async () => {
    setProblem(null)
    const ids = picked.slice(0, MAX_COMPARE)
    let audio: Blob | null = null
    const ctrl = new AbortController()
    abort.current?.abort()
    abort.current = ctrl
    try {
      if (task === 'stt') {
        if (source === UPLOAD) {
          if (!file) {
            setProblem('Choose an audio file, or pick one of the samples.')
            return
          }
          audio = file
        } else {
          const s = audioSamples.find((x) => x.id === source)
          if (!s?.url) return
          audio = new Blob([await api.bytes(s.url, ctrl.signal)], { type: 'audio/wav' })
        }
      } else if (!text.trim()) {
        setProblem('Type something to say.')
        return
      }
    } catch (err) {
      setProblem(describeError(err))
      return
    }
    revoke()
    setRunning(true)
    setRanTask(task)
    setResults(ids.map((model) => ({ model, status: 'running' })))
    const settle = (r: CompareResult) => {
      if (r.audioUrl) urls.current.push(r.audioUrl)
      if (!ctrl.signal.aborted) setResults((prev) => prev.map((p) => (p.model === r.model ? r : p)))
    }
    // All at once: that is how a comparison under equal conditions has to run, and each result shows as it lands.
    await Promise.allSettled(
      ids.map((model) =>
        (task === 'stt' && audio
          ? compareTranscription(model, audio, language.trim() || undefined, ctrl.signal)
          : compareSpeech(model, { input: text, language: language.trim() || undefined }, ctrl.signal)
        ).then(settle),
      ),
    )
    if (!ctrl.signal.aborted) setRunning(false)
  }

  if (isDemo) {
    return (
      <Page title="Compare">
        <NeedsServer title="Comparing models needs a bolo server">
          This screen sends the same audio or text to several of a server&apos;s models, local and cloud, and shows
          their output, latency and estimated cost side by side. This demo runs in your browser only.
        </NeedsServer>
      </Page>
    )
  }

  const best = running ? null : fastest(results)
  const ready = picked.length >= MIN_COMPARE

  return (
    <Page title="Compare">
      <p className="mb-4 max-w-3xl text-[13px] leading-5 text-muted-foreground">
        Send the same audio or the same text to two or more models at once, local and cloud, and read their output,
        latency and estimated cost side by side.
      </p>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] xl:items-start">
        <div className="grid min-w-0 gap-4">
          <Panel
            title="What to compare"
            actions={
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={task}
                onValueChange={(v) => v && setTask(v as Task)}
                aria-label="Task"
              >
                <ToggleGroupItem value="stt" className="px-2.5 text-xs">
                  Speech to text
                </ToggleGroupItem>
                <ToggleGroupItem value="tts" className="px-2.5 text-xs">
                  Text to speech
                </ToggleGroupItem>
              </ToggleGroup>
            }
          >
            <div className="grid gap-4">
              {task === 'stt' ? (
                <div className="grid gap-1.5">
                  <Label htmlFor={`${id}-source`} className="text-xs">
                    Audio
                  </Label>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select
                      value={source}
                      onValueChange={(v) => {
                        setSample(v)
                        if (v === UPLOAD) fileInput.current?.click()
                      }}
                    >
                      <SelectTrigger id={`${id}-source`} className="h-8 min-w-48 flex-1" aria-label="Audio">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent position="popper" align="start">
                        {audioSamples.map((s) => (
                          <SelectItem key={s.id} value={s.id}>
                            Sample: {s.id}
                            {s.seconds ? ` (${fmtSeconds(s.seconds)})` : ''}
                          </SelectItem>
                        ))}
                        <SelectItem value={UPLOAD}>{file ? `File: ${file.name}` : 'A file of yours…'}</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button type="button" size="sm" variant="outline" onClick={() => fileInput.current?.click()}>
                      <FileUp /> Choose file
                    </Button>
                    <input
                      ref={fileInput}
                      type="file"
                      accept="audio/*,video/*,.wav,.mp3,.flac,.ogg,.opus,.m4a,.webm"
                      className="sr-only"
                      tabIndex={-1}
                      aria-label="Audio file to compare"
                      onChange={(e) => {
                        const f = e.target.files?.[0]
                        if (f) {
                          setFile(f)
                          setSample(UPLOAD)
                        }
                        e.target.value = ''
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="grid gap-1.5">
                  <Label htmlFor={`${id}-text`} className="text-xs">
                    Text
                  </Label>
                  <Textarea
                    id={`${id}-text`}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="min-h-24 resize-y text-[14px] leading-6"
                    placeholder="Type something to say…"
                  />
                  <p className="text-xs text-muted-foreground">Each model speaks with its default voice.</p>
                </div>
              )}
              <div className="grid gap-1.5">
                <Label htmlFor={`${id}-lang`} className="text-xs">
                  Language <span className="font-normal text-muted-foreground">(optional)</span>
                </Label>
                <Input
                  id={`${id}-lang`}
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  placeholder="hi, en, ta-IN…"
                  className="h-8 max-w-40 num"
                  autoComplete="off"
                />
              </div>
            </div>
          </Panel>
          <Panel
            title="Models"
            description={`${picked.length} chosen · ${MIN_COMPARE} to ${MAX_COMPARE}`}
            data-testid="compare-models"
          >
            {catalog.isLoading ? (
              <Skeleton className="h-24" />
            ) : (
              <ModelChoice
                models={models}
                value={picked}
                onChange={(ids) => setChosen((c) => ({ ...c, [task]: ids }))}
              />
            )}
          </Panel>
          {cloudPicked.length > 0 && (
            <Alert data-testid="compare-cloud-notice">
              <Cloud />
              <AlertTitle>{CLOUD_NOTICE}</AlertTitle>
              <AlertDescription>
                Running sends this {task === 'stt' ? 'audio' : 'text'} to{' '}
                {[...new Set(cloudPicked.map((x) => byId.get(x)?.provider?.name ?? x))].join(', ')} and is billed by{' '}
                {cloudPicked.length === 1 ? 'that provider' : 'those providers'}.
              </AlertDescription>
            </Alert>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {running ? (
              <Button onClick={stop} variant="outline">
                <Square className="fill-current" /> Stop
              </Button>
            ) : (
              <Button onClick={() => void run()} disabled={!ready} data-testid="compare-run">
                <Play className="fill-current" /> Compare
              </Button>
            )}
            {running && (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground" role="status">
                <Loader2 className="size-3.5 animate-spin" aria-hidden /> Waiting for{' '}
                {results.filter((r) => r.status === 'running').length} of {results.length}
              </span>
            )}
            {!ready && !running && (
              <span className="text-xs text-muted-foreground">Choose at least {MIN_COMPARE} models.</span>
            )}
            {problem && (
              <span className="text-xs text-critical-text" role="alert">
                {problem}
              </span>
            )}
          </div>
        </div>
        <div className="min-w-0" aria-live="polite">
          {results.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border bg-card">
              <Empty icon={Columns3} title="Nothing compared yet">
                Pick the models, then press Compare. The requests run at the same time; times come from each
                request&apos;s trace on the server.
              </Empty>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2" data-testid="compare-results">
              {results.map((r) => (
                <ResultCard key={r.model} result={r} model={byId.get(r.model)} task={ranTask} best={best === r.model} />
              ))}
            </div>
          )}
        </div>
      </div>
    </Page>
  )
}
