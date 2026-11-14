import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CircleAlert, CircleCheck, CircleDashed, Cloud, Gauge, RefreshCw, Sparkles, WandSparkles } from 'lucide-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { describeError, useCatalog, useHardware, useModelActions } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { CatalogModel, ModelFit, Recommendation, SetupUse } from '@/lib/api/types'
import {
  FIT_LABEL,
  fitLine,
  fitMemory,
  fitRtf,
  fitTone,
  hardwareRows,
  isDefaultPick,
  isRecommended,
  speedLine,
} from '@/lib/fit'
import { fmtBytes } from '@/lib/format'
import { groupLanguages, languageLabel } from '@/lib/languages'
import { cn } from '@/lib/utils'

const TONE_CLASS = { good: 'text-good-text', warning: 'text-warning-text', neutral: 'text-muted-foreground' } as const

function FitIcon({ level, className }: { level: string; className?: string }) {
  const Icon =
    level === 'cloud'
      ? Cloud
      : fitTone(level) === 'good'
        ? CircleCheck
        : fitTone(level) === 'warning'
          ? CircleAlert
          : CircleDashed
  return <Icon className={cn('size-3.5 shrink-0', className)} aria-hidden />
}

/** "Recommended for this machine": one of the few the server marks for the model's task (and language). */
export function RecommendedBadge({
  m,
  lang,
  className,
}: {
  m: Pick<CatalogModel, 'fit'>
  lang?: string | null
  className?: string
}) {
  if (!isRecommended(m, lang)) return null
  const first = isDefaultPick(m, lang)
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex h-5 items-center gap-1 rounded-md border border-good/40 bg-good/10 px-1.5 text-[11px] font-medium text-good-text',
            className,
          )}
          data-testid="recommended-badge"
        >
          <Sparkles className="size-3" aria-hidden />
          Recommended for this machine
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {first ? 'The default pick' : 'One of the few picks'} for its task on the server's hardware. A suggestion: every
        other model can still be pulled and loaded.
      </TooltipContent>
    </Tooltip>
  )
}

/** One line on a card: how the model is expected to run on the server's machine. The reason is the tooltip. */
export function FitLine({ fit, className }: { fit: ModelFit | null | undefined; className?: string }) {
  if (!fit) return null
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn('inline-flex min-w-0 items-center gap-1 text-xs', TONE_CLASS[fitTone(fit.level)], className)}
          data-testid="fit-line"
          data-level={fit.level}
        >
          <FitIcon level={fit.level} />
          <span className="truncate">{fitLine(fit)}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">{fit.reason}</TooltipContent>
    </Tooltip>
  )
}

/** The details panel's "On this machine" section, with the notice for models that are unlikely to run well. */
export function FitSection({ m, lang }: { m: CatalogModel; lang?: string | null }) {
  const fit = m.fit
  if (!fit) return null
  const tone = fitTone(fit.level)
  const rtf = fitRtf(fit)
  const memory = fitMemory(fit)
  return (
    <section data-testid="fit-section">
      <h3 className="mb-2 eyebrow">On this machine</h3>
      <div className="rounded-lg border border-border p-3 text-[13px] leading-5">
        <p className={cn('flex flex-wrap items-center gap-x-2 gap-y-1 font-medium', TONE_CLASS[tone])}>
          <span className="inline-flex items-center gap-1.5">
            <FitIcon level={fit.level} />
            {FIT_LABEL[fit.level] ?? fit.label ?? fit.level}
          </span>
          <RecommendedBadge m={m} lang={lang} />
        </p>
        <p className="mt-1.5 text-muted-foreground" data-testid="fit-reason">
          {fit.reason}
        </p>
        {(rtf || memory) && (
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
            {rtf && (
              <>
                <dt className="text-muted-foreground">Expected real-time factor</dt>
                <dd className="text-right num">{rtf}</dd>
              </>
            )}
            {memory && (
              <>
                <dt className="text-muted-foreground">Needs</dt>
                <dd className="text-right">{memory}</dd>
              </>
            )}
          </dl>
        )}
        {rtf && (
          <p className="mt-1 text-xs text-muted-foreground" data-testid="fit-basis">
            {fit.rtf_basis === 'measured' ? 'Measured by the maintainers' : 'Estimated from a similar model'} on the
            reference machine,{' '}
            {fit.scaled_by === 'benchmark' ? 'scaled by the benchmark' : 'roughly scaled to this CPU'}. Once the model
            is loaded, "Measured on this server" below shows the real figure.
          </p>
        )}
        {(fit.notes?.length ?? 0) > 0 && (
          <ul className="mt-2 list-disc pl-4 text-xs text-muted-foreground">
            {fit.notes?.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
      </div>
      {tone === 'warning' && (
        <p
          className="mt-2 rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5"
          role="note"
          data-testid="fit-notice"
        >
          {fit.level === 'unlikely'
            ? 'This model is unlikely to fit or to run well on this machine.'
            : 'This model is expected to be slower than real time on this machine.'}{' '}
          That is a recommendation, not a restriction: you can still pull and load it.
        </p>
      )}
    </section>
  )
}

/** What the server runs on, how fast it is against the reference machine, and the way into "Pick for me". */
export function HardwareCard({ className, onPick }: { className?: string; onPick?: () => void }) {
  const hw = useHardware()
  const qc = useQueryClient()
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ['hardware'] })
    void qc.invalidateQueries({ queryKey: ['catalog'] })
    void qc.invalidateQueries({ queryKey: ['setup'] })
  }
  const benchmark = useMutation({
    mutationFn: () => api.benchmarkHardware(),
    onSuccess: (res) => {
      toast.success('Benchmark stored', {
        description: `This machine runs at ${res.speed.factor.toFixed(2)}× the reference machine. Expected speeds now use it.`,
      })
      refresh()
    },
    onError: (err) => toast.error('The benchmark did not run', { description: describeError(err) }),
  })
  const again = useMutation({
    mutationFn: () => api.hardware(true),
    onSuccess: (data) => {
      qc.setQueryData(['hardware'], data)
      void qc.invalidateQueries({ queryKey: ['catalog'] })
    },
    onError: (err) => toast.error('Could not look again', { description: describeError(err) }),
  })
  if (!hw.offered || hw.isError) return null
  const data = hw.data
  return (
    <Panel
      title="This machine"
      description="What the server runs on"
      className={className}
      data-testid="hardware-card"
      actions={
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={() => again.mutate()}
          disabled={again.isPending || !data}
          aria-label="Detect the hardware again"
          title="Detect the hardware again"
        >
          <RefreshCw className={cn(again.isPending && 'animate-spin')} />
        </Button>
      }
    >
      {!data ? (
        <Skeleton className="h-16 w-full" />
      ) : (
        <>
          <dl className="grid gap-x-6 gap-y-1.5 text-[13px] sm:grid-cols-[auto_1fr]">
            {hardwareRows(data).map((r) => (
              <div key={r.label} className="contents">
                <dt className="text-muted-foreground">{r.label}</dt>
                <dd className="min-w-0 break-words">{r.value}</dd>
              </div>
            ))}
            <div className="contents">
              <dt className="text-muted-foreground">Speed</dt>
              <dd className="min-w-0" data-testid="hardware-speed">
                {speedLine(data)}
                {data.speed.source !== 'benchmark' && (
                  <span className="text-muted-foreground">
                    {' '}
                    · expected speeds are rougher until the benchmark has run
                  </span>
                )}
              </dd>
            </div>
          </dl>
          {data.notes.length > 0 && (
            <ul className="mt-2 list-disc pl-4 text-xs leading-5 text-muted-foreground">
              {data.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
            {onPick && (
              <Button size="sm" onClick={onPick} data-testid="pick-for-me">
                <WandSparkles /> Pick for me
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              onClick={() => benchmark.mutate()}
              disabled={benchmark.isPending}
              title="A few seconds of CPU on the server: times a tiny fixed workload so expected speeds are scaled by a measurement"
              data-testid="run-benchmark"
            >
              <Gauge />{' '}
              {benchmark.isPending
                ? 'Measuring…'
                : data.speed.source === 'benchmark'
                  ? 'Benchmark again'
                  : 'Run benchmark'}
            </Button>
            <p className="min-w-0 flex-1 basis-64 text-xs leading-5 text-muted-foreground">
              This guides the recommendations below; it never restricts what you can pull or load.
            </p>
          </div>
        </>
      )}
    </Panel>
  )
}

const USE_OPTIONS: { value: SetupUse; label: string; hint: string }[] = [
  {
    value: 'live',
    label: 'Live conversation',
    hint: 'a voice agent or live captions: models that keep up with speech',
  },
  {
    value: 'files',
    label: 'Transcribing files',
    hint: 'recordings and meetings: the most capable model that keeps up',
  },
  { value: 'voiceover', label: 'Voice-over', hint: 'narration from text: the best voice that keeps up' },
]

const PICK_TASK: Record<string, string> = {
  stt: 'Speech to text',
  tts: 'Text to speech',
  diarization: 'Speaker diarization',
  vad: 'Voice-activity detection',
  turn: 'End-of-turn detection',
}

function PickRow({
  task,
  optional,
  model,
  alternatives,
  note,
  checked,
  onChecked,
}: {
  task: string
  optional: boolean
  model: Recommendation | null
  alternatives: Recommendation[]
  note: string | null
  checked: boolean
  onChecked: (v: boolean) => void
}) {
  return (
    <li className="flex items-start gap-3 py-2.5" data-testid={`pick-${task}`}>
      <Checkbox
        className="mt-0.5"
        checked={model != null && !model.installed && checked}
        disabled={model == null || model.installed}
        onCheckedChange={(v) => onChecked(v === true)}
        aria-label={`Pull ${model?.id ?? task}`}
      />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 text-xs text-muted-foreground">
          {PICK_TASK[task] ?? task}
          {optional && <span>· optional</span>}
        </div>
        {model ? (
          <>
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="num text-sm font-medium">{model.id}</span>
              <span className="text-xs text-muted-foreground">
                {model.installed ? 'installed' : fmtBytes(model.size_bytes)} · {model.license}
                {!model.commercial_use && <span className="text-warning-text"> · non-commercial</span>}
              </span>
            </div>
            <FitLine fit={model.fit} className="mt-0.5" />
            <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{model.fit.reason}</p>
            {alternatives.length > 0 && (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Also suitable: <span className="num">{alternatives.map((a) => a.id).join(', ')}</span>
              </p>
            )}
          </>
        ) : (
          <p className="text-[13px] text-muted-foreground">{note ?? 'Nothing in the catalog suits this.'}</p>
        )}
      </div>
    </li>
  )
}

/** "Pick for me": proposes a model per task for a language and a use, and pulls the chosen ones in one click. */
export function PickForMe({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const catalog = useCatalog()
  const actions = useModelActions()
  const [use, setUse] = useState<SetupUse>('live')
  const [language, setLanguage] = useState('en')
  const [commercial, setCommercial] = useState(false)
  const [skipped, setSkipped] = useState<Record<string, boolean>>({})
  const languages = useMemo(() => {
    const codes = (catalog.data?.models ?? [])
      .filter((m) => m.task === 'stt' || m.task === 'tts')
      .flatMap((m) => m.languages)
    const g = groupLanguages(codes)
    return [...g.english, ...g.indic, ...g.other]
  }, [catalog.data])
  const plan = useQuery({
    queryKey: ['setup', use, language, commercial],
    queryFn: ({ signal }) => api.recommendSetup({ use, language, commercial: commercial || undefined }, signal),
    enabled: open,
    retry: false,
  })
  const toPull = (plan.data?.picks ?? [])
    .map((p) => p.model)
    .filter((m): m is Recommendation => m != null && !m.installed && !m.cloud && !skipped[m.id])
  const pullAll = () => {
    for (const m of toPull) actions.pull.mutate(m.id)
    onOpenChange(false)
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl" data-testid="pick-dialog">
        <DialogHeader>
          <DialogTitle>Pick models for this machine</DialogTitle>
          <DialogDescription>
            A starting set for one language and one use, chosen for the server's hardware. Nothing is pulled until you
            say so, and every other model stays available.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={use}
            onValueChange={(v) => v && setUse(v as SetupUse)}
            aria-label="What for"
            className="flex-wrap justify-start"
          >
            {USE_OPTIONS.map((o) => (
              <ToggleGroupItem key={o.value} value={o.value} className="px-2.5 text-xs" title={o.hint}>
                {o.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="h-8 w-52" aria-label="Language">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start" className="max-h-72">
                {(languages.includes(language) ? languages : [language, ...languages]).map((l) => (
                  <SelectItem key={l} value={l}>
                    {languageLabel(l)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <label className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30">
              Commercial use only
              <Switch checked={commercial} onCheckedChange={setCommercial} size="sm" />
            </label>
          </div>
          {plan.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : plan.isError ? (
            <p className="text-[13px] text-muted-foreground" role="alert">
              Could not get a proposal: {describeError(plan.error)}
            </p>
          ) : plan.data ? (
            <>
              <p className="text-xs leading-5 text-muted-foreground" data-testid="pick-hardware">
                For {plan.data.hardware}. Expected speeds are{' '}
                {plan.data.speed.source === 'benchmark'
                  ? 'scaled by the stored benchmark.'
                  : 'a rough estimate until the benchmark has run.'}
              </p>
              <ul className="divide-y divide-border rounded-lg border border-border px-3">
                {plan.data.picks.map((p) => (
                  <PickRow
                    key={p.task}
                    {...p}
                    checked={p.model != null && !skipped[p.model.id]}
                    onChecked={(v) => p.model && setSkipped((s) => ({ ...s, [p.model?.id ?? '']: !v }))}
                  />
                ))}
              </ul>
            </>
          ) : null}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Not now
          </Button>
          <Button onClick={pullAll} disabled={toPull.length === 0} data-testid="pick-pull">
            {toPull.length === 0
              ? 'Nothing to pull'
              : `Pull ${toPull.length} model${toPull.length === 1 ? '' : 's'} (${fmtBytes(toPull.reduce((a, m) => a + m.size_bytes, 0))})`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Shown while no model is installed yet: the shortest way from an empty server to a working one. */
export function FirstRunPick({
  onPick,
  onDismiss,
  className,
}: {
  onPick: () => void
  onDismiss: () => void
  className?: string
}) {
  return (
    <div
      className={cn('flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card p-4', className)}
      data-testid="first-run-pick"
    >
      <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/50 text-muted-foreground">
        <WandSparkles className="size-4" aria-hidden />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="text-sm font-semibold">No models installed yet</h2>
        <p className="text-[13px] leading-5 text-muted-foreground">
          Tell the console your language and what you want to do; it proposes speech-to-text and text-to-speech models
          that suit this machine and pulls them in one click. Or choose any model below yourself.
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button size="sm" variant="ghost" onClick={onDismiss}>
          Not now
        </Button>
        <Button size="sm" onClick={onPick} data-testid="first-run-pick-button">
          <WandSparkles /> Pick for me
        </Button>
      </div>
    </div>
  )
}
