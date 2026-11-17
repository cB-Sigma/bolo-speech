import { FileAudio, Play, Upload, Users } from 'lucide-react'
import { useId, useMemo, useRef, useState } from 'react'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { describeError } from '@/hooks/use-api'
import { api, ApiError } from '@/lib/api/client'
import type { CatalogModel, Sample, ServerInfo } from '@/lib/api/types'
import { diarizerOptions, separationModels, speakerConditionedModels } from '@/lib/capabilities'
import { speakerCountProblem, wavSampleRate, type DiarizedResult, type SpeakerCount } from '@/lib/diarize'
import { cn } from '@/lib/utils'
import { SpeakerResult } from './speaker-result'

const PAGE_MODEL = '__page'

interface Picked {
  name: string
  blob: Blob
  /** Sample rate when the file says (WAV); null otherwise. */
  rate: number | null
}

/**
 * Transcribe a recording with speaker labels: diarizer choice, speaker count
 * hints, speaker-conditioned recognisers (DiCoW) and separation of overlapped
 * speech, then the speaker-attributed result. Server only.
 */
export function SpeakersPanel({
  info,
  catalog,
  model,
  language,
  samples,
  className,
}: {
  info: ServerInfo | undefined
  catalog: CatalogModel[]
  /** The model chosen on the page. */
  model: CatalogModel | null
  language: string
  samples: Sample[]
  className?: string
}) {
  const ids = useId()
  const fileInput = useRef<HTMLInputElement>(null)
  const diarizers = useMemo(() => diarizerOptions(info, catalog), [info, catalog])
  const conditioned = useMemo(() => speakerConditionedModels(info, catalog), [info, catalog])
  const separators = useMemo(() => separationModels(catalog), [catalog])
  const sepCap = info?.capabilities?.separation
  const minRate = sepCap?.min_sample_rate ?? 16000

  const [file, setFile] = useState<Picked | null>(null)
  const [diarize, setDiarize] = useState(true)
  const [diarizer, setDiarizer] = useState<string>('')
  const [countMode, setCountMode] = useState<SpeakerCount['mode']>('auto')
  const [exact, setExact] = useState('2')
  const [min, setMin] = useState('')
  const [max, setMax] = useState('')
  const [recogniser, setRecogniser] = useState(PAGE_MODEL)
  const [separate, setSeparate] = useState(false)
  const [separator, setSeparator] = useState('')
  const [narrowband, setNarrowband] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DiarizedResult | null>(null)
  const [error, setError] = useState<{ message: string; param: string | null } | null>(null)

  const defaultDiarizer = diarizers.find((d) => d.isDefault)?.id ?? diarizers[0]?.id ?? ''
  const chosenDiarizer = diarizer || defaultDiarizer
  const conditionedModel = conditioned.find((m) => m.id === recogniser) ?? null
  const usedModel = conditionedModel ?? model
  const count: SpeakerCount =
    countMode === 'exact'
      ? { mode: 'exact', n: Number(exact) }
      : countMode === 'range'
        ? { mode: 'range', min: min ? Number(min) : undefined, max: max ? Number(max) : undefined }
        : { mode: 'auto' }
  const countProblem = diarize ? speakerCountProblem(count) : null
  const lowRate = file?.rate != null && file.rate < minRate
  const separationBlocked = !diarize
    ? 'Needs speaker labels.'
    : conditionedModel
      ? 'Speaker-conditioned models already transcribe overlapping speech per speaker.'
      : separators.length === 0
        ? 'No separation model in the catalog.'
        : lowRate && !narrowband
          ? `Recorded at ${file?.rate} Hz: separation is trained on ${minRate / 1000} kHz audio and above.`
          : null
  const separateOn = separate && !separationBlocked
  const chosenSeparator = separator || sepCap?.default_model || separators[0]?.id || ''

  const pick = async (name: string, blob: Blob) => {
    const head = await blob.slice(0, 4096).arrayBuffer()
    setFile({ name, blob, rate: wavSampleRate(head) })
    setResult(null)
    setError(null)
  }
  const pickSample = async (s: Sample) => {
    if (!s.url) return
    try {
      const bytes = await api.bytes(s.url)
      await pick(`${s.id}.wav`, new Blob([bytes], { type: 'audio/wav' }))
    } catch (err) {
      setError({ message: describeError(err), param: null })
    }
  }

  const run = async () => {
    if (!file || !usedModel) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.transcribeSegments(file.blob, file.name, {
        model: usedModel.id,
        language,
        diarize,
        // A speaker-conditioned model brings its own diarizer unless one is named explicitly.
        diarizer: diarize && (diarizer || !conditionedModel) ? chosenDiarizer || undefined : undefined,
        speakers: count,
        separate: separateOn,
        separationModel: separateOn ? chosenSeparator || undefined : undefined,
        allowNarrowband: separateOn && lowRate && narrowband,
      })
      setResult(r)
    } catch (err) {
      const body = err instanceof ApiError ? (err.body as { error?: { param?: string } } | undefined) : undefined
      setError({ message: describeError(err), param: body?.error?.param ?? null })
    } finally {
      setBusy(false)
    }
  }

  const audioSamples = samples.filter((s) => s.kind === 'audio' && s.url && s.id !== 'tone-bursts')
  const note = diarizers.find((d) => d.id === chosenDiarizer)?.note

  return (
    <Panel
      title="Who spoke when"
      description="Transcribe a recording with speaker labels"
      className={className}
      data-testid="speakers-panel"
    >
      <div className="grid gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()} disabled={busy}>
            <Upload /> Choose a recording
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
            className="sr-only"
            tabIndex={-1}
            aria-label="Recording to transcribe with speaker labels"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void pick(f.name, f)
              e.target.value = ''
            }}
          />
          {audioSamples.map((s) => (
            <Button
              key={s.id}
              size="sm"
              variant={file?.name === `${s.id}.wav` ? 'secondary' : 'ghost'}
              className="border border-border"
              onClick={() => void pickSample(s)}
              disabled={busy}
            >
              <FileAudio /> <span className="num text-xs">{s.id}</span>
            </Button>
          ))}
          {file && (
            <span className="min-w-0 truncate text-xs text-muted-foreground" data-testid="speakers-file">
              {file.name}
              {file.rate != null && <span className="num"> · {file.rate / 1000} kHz</span>}
            </span>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="grid gap-1.5">
            <Label htmlFor={`${ids}-rec`} className="text-xs">
              Recogniser
            </Label>
            <Select value={recogniser} onValueChange={setRecogniser} disabled={busy}>
              <SelectTrigger id={`${ids}-rec`} className="h-8 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                <SelectItem value={PAGE_MODEL}>
                  {model ? `${model.id} (chosen above)` : 'Model chosen above'}
                </SelectItem>
                {conditioned.map((m) => (
                  <SelectItem key={m.id} value={m.id} disabled={!m.installed || m.available === false}>
                    {m.id} · speaker-conditioned
                    {m.available === false ? ' · needs a plugin' : !m.installed ? ' · not installed' : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {conditioned.length > 0 && (
              <p className="text-xs leading-4 text-muted-foreground">
                Speaker-conditioned models (DiCoW) transcribe each speaker separately, overlaps included; several times
                slower on CPU.
              </p>
            )}
          </div>

          <div className="grid gap-1.5">
            <div className="flex items-center justify-between gap-2">
              <Label htmlFor={`${ids}-diar`} className="text-xs">
                Speaker labels
              </Label>
              <Switch
                checked={diarize}
                onCheckedChange={setDiarize}
                size="sm"
                disabled={busy}
                aria-label="Label speakers (diarization)"
                data-testid="diarize-toggle"
              />
            </div>
            <Select value={chosenDiarizer} onValueChange={setDiarizer} disabled={busy || !diarize || !diarizers.length}>
              <SelectTrigger id={`${ids}-diar`} className="h-8 w-full" aria-label="Diarizer" data-testid="diarizer">
                <SelectValue placeholder="Server default diarizer" />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                {diarizers.map((d) => (
                  <SelectItem key={d.id} value={d.id} disabled={!d.usable} textValue={d.label}>
                    {d.label}
                    <span className="text-muted-foreground [[data-slot=select-trigger]_&]:hidden">
                      {d.license ? ` · ${d.license}` : ''}
                      {d.runtime ? ` · ${d.runtime}` : ''}
                      {d.gated && <span className="text-warning-text"> · gated</span>}
                      {d.commercialUse === false && <span className="text-warning-text"> · non-commercial</span>}
                      {!d.usable ? ' · needs a component' : !d.installed ? ' · not installed' : ''}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {diarize && note && <p className="text-xs leading-4 text-muted-foreground">{note}</p>}
            {diarize && conditionedModel && !diarizer && (
              <p className="text-xs leading-4 text-muted-foreground">
                {conditionedModel.id} uses nemotron-3-diarization unless you pick a diarizer here.
              </p>
            )}
          </div>
        </div>

        <fieldset className={cn('grid gap-2', !diarize && 'opacity-60')} disabled={!diarize || busy}>
          <legend className="mb-1 text-xs font-medium">Number of speakers</legend>
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={countMode}
              onValueChange={(v) => v && setCountMode(v as SpeakerCount['mode'])}
              aria-label="How many speakers"
            >
              <ToggleGroupItem value="auto" className="px-2.5 text-xs">
                Find out
              </ToggleGroupItem>
              <ToggleGroupItem value="exact" className="px-2.5 text-xs">
                Exactly
              </ToggleGroupItem>
              <ToggleGroupItem value="range" className="px-2.5 text-xs">
                Between
              </ToggleGroupItem>
            </ToggleGroup>
            {countMode === 'exact' && (
              <Input
                type="number"
                min={1}
                max={32}
                value={exact}
                onChange={(e) => setExact(e.target.value)}
                className="h-8 w-20 num"
                aria-label="Number of speakers"
              />
            )}
            {countMode === 'range' && (
              <>
                <Input
                  type="number"
                  min={1}
                  max={32}
                  value={min}
                  placeholder="min"
                  onChange={(e) => setMin(e.target.value)}
                  className="h-8 w-20 num"
                  aria-label="At least this many speakers"
                />
                <span className="text-xs text-muted-foreground">to</span>
                <Input
                  type="number"
                  min={1}
                  max={32}
                  value={max}
                  placeholder="max"
                  onChange={(e) => setMax(e.target.value)}
                  className="h-8 w-20 num"
                  aria-label="At most this many speakers"
                />
              </>
            )}
          </div>
          {countProblem && <p className="text-xs text-critical-text">{countProblem}</p>}
          {chosenDiarizer.startsWith('nemotron') && countMode !== 'auto' && (
            <p className="text-xs leading-4 text-muted-foreground">
              Nemotron can merge the quietest speakers to meet a maximum, but cannot find more speakers than it hears.
            </p>
          )}
        </fieldset>

        <div className="grid gap-2 rounded-lg border border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <Label htmlFor={`${ids}-sep`} className="text-[13px]">
              Separate overlapping speech
            </Label>
            <Switch
              id={`${ids}-sep`}
              checked={separateOn}
              onCheckedChange={setSeparate}
              disabled={busy || separationBlocked != null}
              size="sm"
              data-testid="separate-toggle"
            />
          </div>
          <p className="text-xs leading-4 text-muted-foreground">
            {separationBlocked ??
              `Where two people talk at once, extract each voice and transcribe it on its own (opt-in; ${sepCap?.languages?.join(', ') || 'English'}; adds time).`}
          </p>
          {separators.length > 0 && (
            <Select value={chosenSeparator} onValueChange={setSeparator} disabled={busy || !separateOn}>
              <SelectTrigger className="h-8 w-full sm:w-72" aria-label="Separation model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                {separators.map((m) => (
                  <SelectItem key={m.id} value={m.id} disabled={m.available === false}>
                    {m.id}
                    {m.id === sepCap?.default_model ? ' (default)' : ''}
                    {!m.installed ? ' · not installed' : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {(lowRate || error?.param === 'separate_overlaps') && diarize && !conditionedModel && (
            <label className="flex items-center gap-2 text-xs">
              <Checkbox
                checked={narrowband}
                onCheckedChange={(v) => setNarrowband(v === true)}
                aria-label="Separate telephone-band audio anyway"
              />
              Separate telephone-band audio anyway (not tuned for it)
            </label>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            onClick={() => void run()}
            disabled={!file || !usedModel?.installed || busy || countProblem != null}
            data-testid="speakers-run"
          >
            {busy ? <Users className="live-dot" /> : <Play className="fill-current" />}
            {busy ? 'Transcribing…' : diarize ? 'Transcribe with speakers' : 'Transcribe'}
          </Button>
          {!file && <span className="text-xs text-muted-foreground">Choose a recording or a sample first.</span>}
          {file && usedModel && !usedModel.installed && (
            <span className="text-xs text-muted-foreground">{usedModel.id} is not installed.</span>
          )}
        </div>

        {error && (
          <p role="alert" className="rounded-lg border border-critical/30 bg-critical/10 p-3 text-[13px] leading-5">
            {error.message}
          </p>
        )}
        {result && <SpeakerResult result={result} />}
      </div>
    </Panel>
  )
}
