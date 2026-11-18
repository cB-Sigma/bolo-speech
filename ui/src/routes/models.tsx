import {
  Boxes,
  Cloud,
  Download,
  ExternalLink,
  LayoutGrid,
  MoreHorizontal,
  Play,
  Power,
  PowerOff,
  Rows3,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Page } from '@/components/layout/app-shell'
import {
  AvailabilityMark,
  CloudMark,
  EngineMissingMark,
  GatedMark,
  LicenseNoteMark,
  RemoteCodeMark,
  HardwareMark,
  languageSummary,
  StateDot,
  TIER_LABEL,
  TierBadge,
  TIERS,
  ValidatedMark,
} from '@/components/model-bits'
import { Empty, ErrorState, Readout } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  ComponentInstallButton,
  ComponentNotice,
  CopyCommand,
  ComponentsCard,
  NeedsComponentMark,
} from '@/components/component-install'
import { FirstRunPick, FitLine, FitSection, HardwareCard, PickForMe, RecommendedBadge } from '@/components/fit-bits'
import { DownloadsPanel } from '@/components/downloads-panel'
import { LoadedModelsPanel } from '@/components/loaded-models'
import { ModelSettingsSection } from '@/components/model-settings'
import {
  cloudNotice,
  engineMissing,
  isCloud,
  isCpuFriendly,
  isNotImplemented,
  isPending,
  PHONEMIZER_LABEL,
  remoteCodeCommand,
  runtimeLabel,
} from '@/lib/capabilities'
import { describePrice } from '@/lib/cost'
import {
  describeError,
  missingComponent,
  useCatalog,
  useComponents,
  useInfo,
  useLive,
  useModelActions,
} from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import type { CatalogModel, ModelStats, Task } from '@/lib/api/types'
import { isUnlikely, sortByFit } from '@/lib/fit'
import { groupLanguages, languageLabel } from '@/lib/languages'
import { fmtBytes, fmtInt, fmtMs, fmtParams, fmtRtf, fmtSpeed, languageName } from '@/lib/format'
import { prefs, setPref } from '@/lib/prefs'
import { NeedsServer, SERVER_ONLY_HINT } from '@/components/demo'
import { isDemo } from '@/lib/api/config'
import { cn } from '@/lib/utils'

const TASK_LABEL: Record<Task, string> = {
  stt: 'Speech to text',
  tts: 'Text to speech',
  vad: 'Voice activity',
  diarization: 'Diarization',
  separation: 'Overlapped speech separation',
  denoise: 'Denoising',
  turn: 'End of turn',
  translation: 'Translation',
  lid: 'Language identification',
  'speaker-embedding': 'Speaker embedding',
  'speech-to-speech': 'Speech to speech',
  'audio-llm': 'Audio-in language models',
  'audio-tagging': 'Sound events',
  emotion: 'Speech emotion',
  'voice-conversion': 'Voice conversion',
  align: 'Forced alignment',
}
const TASK_SHORT: Record<Task, string> = {
  stt: 'STT',
  tts: 'TTS',
  vad: 'VAD',
  diarization: 'Diarization',
  separation: 'Separation',
  denoise: 'Denoise',
  turn: 'Turn',
  translation: 'Translation',
  lid: 'LID',
  'speaker-embedding': 'Embedding',
  'speech-to-speech': 'S2S',
  'audio-llm': 'Audio LLM',
  'audio-tagging': 'Tags',
  emotion: 'Emotion',
  'voice-conversion': 'Convert',
  align: 'Align',
}
const TASK_ORDER: Task[] = [
  'stt',
  'tts',
  'vad',
  'turn',
  'lid',
  'align',
  'diarization',
  'speaker-embedding',
  'separation',
  'denoise',
  'audio-tagging',
  'emotion',
  'voice-conversion',
  'speech-to-speech',
  'audio-llm',
  'translation',
]

/** "Needs eSpeak NG" for a model whose optional component is not installed; nothing otherwise. */
function UnmetMark({ m }: { m: CatalogModel }) {
  const missing = missingComponent(m, useComponents().data?.components)
  return missing ? <NeedsComponentMark name={missing.name} /> : null
}

/** Licence note, remote code and a missing engine, as marks on a card or in the drawer. */
function LicenceMarks({ m }: { m: CatalogModel }) {
  const info = useInfo().data
  const engine = isDemo ? null : engineMissing(info, m)
  return (
    <>
      {m.license_note && <LicenseNoteMark note={m.license_note} />}
      <RemoteCodeMark model={m} />
      {engine && <EngineMissingMark engine={engine.engine} reason={engine.reason} />}
    </>
  )
}

/** The drawer's account of what to do before using a model: licence terms, remote code, a missing engine. */
function LicenceDetails({ m }: { m: CatalogModel }) {
  const info = useInfo().data
  const engine = isDemo ? null : engineMissing(info, m)
  const rc = m.remote_code
  const poor = Object.entries(m.not_recommended_for ?? {})
  if (!m.license_note && !rc && !engine && poor.length === 0) return null
  return (
    <div className="grid min-w-0 gap-3" data-testid="licence-details">
      {engine && (
        <div className="min-w-0 rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5">
          <p className="font-medium text-warning-text">The {engine.engine} engine is not installed on this server</p>
          <p className="mt-1 text-muted-foreground">
            {m.id} can be pulled, but it cannot load until it is. {engine.reason}
          </p>
          {engine.install && <CopyCommand command={engine.install} className="mt-2" label="Install command" />}
        </div>
      )}
      {poor.length > 0 && (
        <div className="rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5">
          <p className="font-medium text-warning-text">Not recommended for some languages</p>
          <ul className="mt-1 grid gap-1 text-muted-foreground" data-testid="not-recommended-for">
            {poor.map(([code, why]) => (
              <li key={code}>
                <span className="font-medium text-foreground">{languageName(code)}</span>: {why}
              </li>
            ))}
          </ul>
        </div>
      )}
      {m.license_note && (
        <div className="rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5">
          <p className="font-medium text-warning-text">Licence note</p>
          <p className="mt-1 text-muted-foreground" data-testid="license-note">
            {m.license_note}
          </p>
        </div>
      )}
      {rc && (
        <div
          className={cn(
            'min-w-0 rounded-lg border p-3 text-[13px] leading-5',
            rc.allowed ? 'border-border' : 'border-warning/40 bg-warning/10',
          )}
          data-testid="remote-code"
          data-allowed={rc.allowed}
        >
          <p className={cn('font-medium', !rc.allowed && 'text-warning-text')}>
            Remote code · {rc.allowed ? 'allowed for this model' : 'not allowed yet'}
          </p>
          {rc.notice && <p className="mt-1 text-muted-foreground">{rc.notice}</p>}
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-muted-foreground">Commit</dt>
            <dd className="num break-all">{rc.revision ?? 'not stated'}</dd>
            <dt className="text-muted-foreground">Files</dt>
            <dd className="num break-all">{rc.files.join(', ') || 'none listed'}</dd>
            {rc.reviewed && (
              <>
                <dt className="text-muted-foreground">Review</dt>
                <dd className="text-muted-foreground">{rc.reviewed}</dd>
              </>
            )}
          </dl>
          {!rc.allowed && <CopyCommand command={remoteCodeCommand(m.id)} className="mt-2" label="Opt-in command" />}
        </div>
      )}
    </div>
  )
}

/** A row under a card: what the model still needs, with an install button when the server can install it. */
function UnmetRow({ m }: { m: CatalogModel }) {
  const missing = missingComponent(m, useComponents().data?.components)
  if (!missing || isDemo) return null
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2">
      <NeedsComponentMark name={missing.name} />
      {missing.installable ? (
        <ComponentInstallButton component={missing} label={`Install ${missing.id}`} className="h-7" />
      ) : (
        <span className="text-xs text-muted-foreground">see Components</span>
      )}
    </div>
  )
}

function UnmetNotice({ m }: { m: CatalogModel }) {
  const missing = missingComponent(m, useComponents().data?.components)
  return missing ? <ComponentNotice component={missing} subject={m.id} /> : null
}

const ANY = '__any'
/** Cards shown per task before "Show all". */
const GROUP_LIMIT = 12

type Actions = ReturnType<typeof useModelActions>

function ModelActions({
  m,
  actions,
  onDelete,
  compact,
}: {
  m: CatalogModel
  actions: Actions
  onDelete: (m: CatalogModel) => void
  compact?: boolean
}) {
  const loading = actions.load.isPending && actions.load.variables === m.id
  const unloading = actions.unload.isPending && actions.unload.variables === m.id
  if (isDemo && m.runtime !== 'browser') {
    return (
      <span className="text-xs text-muted-foreground" title={SERVER_ONLY_HINT}>
        Needs the server
      </span>
    )
  }
  if (!m.installed && (isPending(m) || isNotImplemented(m))) {
    return (
      <span className="text-xs text-muted-foreground" title="This model cannot be pulled yet">
        {isPending(m) ? 'Files not published' : 'Cannot run yet'}
      </span>
    )
  }
  if (isCloud(m) && !m.installed) {
    // Nothing to download: the model is usable once its provider has a key.
    return (
      <Button size="sm" variant="outline" asChild data-testid={`setup-${m.id}`}>
        <Link to="/providers">
          <Cloud /> Set up {m.provider?.name ?? 'provider'}
        </Link>
      </Button>
    )
  }
  if (m.pulling) {
    return (
      <div className={cn('flex items-center gap-2', compact ? 'w-32' : 'w-full')}>
        <Progress value={m.pulling.progress * 100} className="h-1.5 flex-1" aria-label={`Downloading ${m.id}`} />
        <span className="w-9 text-right num text-xs text-muted-foreground">
          {Math.round(m.pulling.progress * 100)}%
        </span>
      </div>
    )
  }
  if (!m.installed) {
    return (
      <Button
        size="sm"
        variant="outline"
        onClick={() => actions.pull.mutate(m.id)}
        disabled={actions.pull.isPending && actions.pull.variables === m.id}
        data-testid={`pull-${m.id}`}
      >
        <Download /> {m.runtime === 'browser' ? 'Download' : 'Pull'}
        <span className="num text-muted-foreground">{fmtBytes(m.size_bytes)}</span>
      </Button>
    )
  }
  return (
    <div className="flex items-center gap-1">
      {m.loaded ? (
        <Button size="sm" variant="outline" onClick={() => actions.unload.mutate(m.id)} disabled={unloading}>
          <PowerOff /> {unloading ? 'Unloading…' : 'Unload'}
        </Button>
      ) : (
        <Button
          size="sm"
          variant="outline"
          onClick={() => actions.load.mutate(m.id)}
          disabled={loading || m.available === false}
          title={m.available === false ? 'Install the component this model needs first' : undefined}
          data-testid={`load-${m.id}`}
        >
          <Power /> {loading ? 'Loading…' : 'Load'}
        </Button>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button size="icon-sm" variant="ghost" aria-label={`More actions for ${m.id}`}>
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <a href={m.source_url} target="_blank" rel="noreferrer">
              <ExternalLink /> Open source page
            </a>
          </DropdownMenuItem>
          {!isCloud(m) && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onSelect={() => onDelete(m)}>
                <Trash2 /> Delete from disk
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

function MemoryBar({ bytes, max }: { bytes: number; max: number }) {
  return (
    <div className="flex items-center gap-2" title={`${fmtBytes(bytes)} resident`}>
      <div
        className="h-1.5 flex-1 overflow-hidden rounded-full bg-chart-1/15"
        role="meter"
        aria-label="Memory in use"
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={bytes}
        aria-valuetext={fmtBytes(bytes)}
      >
        <div className="h-full rounded-full bg-chart-1" style={{ width: `${Math.max(2, (bytes / max) * 100)}%` }} />
      </div>
      <span className="w-16 text-right num text-xs text-muted-foreground">{fmtBytes(bytes)}</span>
    </div>
  )
}

function ModelCard({
  m,
  stats,
  maxMem,
  actions,
  lang,
  onOpen,
  onDelete,
}: {
  m: CatalogModel
  stats?: ModelStats
  maxMem: number
  actions: Actions
  /** The language filter, if one is set: "recommended" is judged for it. */
  lang: string | null
  onOpen: () => void
  onDelete: (m: CatalogModel) => void
}) {
  return (
    <article
      className="flex min-w-0 flex-col rounded-xl border border-border bg-card transition-colors duration-150 hover:border-input"
      data-testid="model-card"
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 flex-col rounded-t-xl px-4 pt-3.5 pb-3 text-left"
      >
        <div className="flex w-full items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate num text-sm font-medium">{m.id}</div>
            <div className="mt-0.5 truncate text-xs text-muted-foreground">
              {m.family} · {TASK_SHORT[m.task] ?? m.task} · {runtimeLabel(m) ?? m.engine}
            </div>
          </div>
          <TierBadge tier={m.tier} className="shrink-0" />
        </div>
        <p className="mt-2 line-clamp-2 min-h-10 text-[13px] leading-5 text-muted-foreground">{m.description}</p>
        <dl className="mt-3 grid w-full grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Languages</dt>
          <dd className="truncate text-right">{languageSummary(m.languages)}</dd>
          <dt className="text-muted-foreground">Size</dt>
          <dd className="text-right num">
            {isCloud(m) ? (
              <span className="text-muted-foreground">no download</span>
            ) : (
              <>
                {fmtBytes(m.size_bytes)}
                <span className="text-muted-foreground"> · {fmtParams(m.params)} params</span>
              </>
            )}
          </dd>
          <dt className="text-muted-foreground">Licence</dt>
          <dd className="truncate text-right">
            {m.license}
            {m.commercial_use === false && <span className="text-warning-text"> · non-commercial</span>}
          </dd>
        </dl>
        <div className="mt-3 flex w-full flex-wrap items-center gap-x-3 gap-y-1">
          <CloudMark model={m} />
          <AvailabilityMark model={m} />
          <ValidatedMark validated={m.validated} />
          {m.gated && <GatedMark />}
          <LicenceMarks m={m} />
          <HardwareMark hardware={m.hardware} />
        </div>
        {m.fit && !isDemo && (
          <div className="mt-2 flex w-full flex-wrap items-center gap-x-3 gap-y-1">
            <RecommendedBadge m={m} lang={lang} />
            <FitLine fit={m.fit} />
          </div>
        )}
      </button>
      <UnmetRow m={m} />
      {m.loaded && stats && stats.memory_bytes > 0 && (
        <div className="px-4 pb-3">
          <MemoryBar bytes={stats.memory_bytes} max={maxMem} />
        </div>
      )}
      <div className="flex min-h-12 items-center justify-between gap-2 border-t border-border px-4 py-2">
        <StateDot model={m} />
        <ModelActions m={m} actions={actions} onDelete={onDelete} compact />
      </div>
    </article>
  )
}

function Details({
  m,
  stats,
  lang,
  onClose,
}: {
  m: CatalogModel | null
  stats?: ModelStats
  lang: string | null
  onClose: () => void
}) {
  const navigate = useNavigate()
  return (
    <Sheet open={m != null} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        className="w-full gap-0 overflow-y-auto sm:max-w-md"
        // Focus the panel itself, not its first tooltip trigger.
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          ;(e.currentTarget as HTMLElement | null)?.focus()
        }}
      >
        {m && (
          <>
            <SheetHeader className="border-b border-border">
              <SheetTitle className="num text-base">{m.id}</SheetTitle>
              <SheetDescription>
                {m.family} · {TASK_LABEL[m.task] ?? m.task}
              </SheetDescription>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <TierBadge tier={m.tier} />
                <StateDot model={m} />
                <CloudMark model={m} />
                <AvailabilityMark model={m} />
                <ValidatedMark validated={m.validated} />
                <UnmetMark m={m} />
                {m.gated && <GatedMark />}
                <LicenceMarks m={m} />
                <HardwareMark hardware={m.hardware} />
              </div>
            </SheetHeader>
            <div className="grid gap-5 p-4 text-sm">
              <p className="leading-6 text-muted-foreground">{m.description}</p>
              <LicenceDetails m={m} />
              {isCloud(m) && (
                <div
                  className="rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5"
                  data-testid="cloud-details"
                >
                  <p className="flex items-center gap-1.5 font-medium text-warning-text">
                    <Cloud className="size-3.5" aria-hidden /> {cloudNotice(m)}
                  </p>
                  <p className="mt-1 text-muted-foreground">
                    Served by the provider <span className="num text-foreground">{m.provider?.name ?? 'unknown'}</span>
                    {m.provider?.model ? (
                      <>
                        {' '}
                        as <span className="num text-foreground">{m.provider.model}</span>
                      </>
                    ) : null}
                    . {m.provider?.configured ? 'The provider is configured.' : 'The provider is not configured yet.'}{' '}
                    {describePrice(m.provider?.price)
                      ? `Your price table: ${describePrice(m.provider?.price)}.`
                      : 'No price is set for it, so no cost estimate is shown.'}{' '}
                    <Link to="/providers" className="underline underline-offset-2">
                      Providers
                    </Link>
                  </p>
                  {(m.fallback?.length ?? 0) > 0 && (
                    <p className="mt-1 text-muted-foreground">
                      Falls back to <span className="num text-foreground">{m.fallback?.join(', ')}</span> when the
                      provider cannot answer.
                    </p>
                  )}
                </div>
              )}
              {!isCloud(m) && (m.fallback?.length ?? 0) > 0 && (
                <p className="text-[13px] leading-5 text-muted-foreground" data-testid="fallback-details">
                  Falls back to <span className="num text-foreground">{m.fallback?.join(', ')}</span> when this model is
                  unavailable or overloaded.
                </p>
              )}
              <UnmetNotice m={m} />
              {!isDemo && <FitSection m={m} lang={lang} />}

              <section>
                <h3 className="mb-2 eyebrow">Measured on this server</h3>
                {m.loaded && stats ? (
                  <div className="grid grid-cols-2 gap-4 rounded-lg border border-border p-3">
                    <Readout
                      label="Real-time factor p50"
                      value={fmtRtf(stats.rtf?.p50)}
                      hint={
                        stats.rtf
                          ? `${fmtSpeed(stats.rtf.p50)} real time · p95 ${fmtRtf(stats.rtf.p95)}`
                          : 'no requests yet'
                      }
                    />
                    <Readout
                      label="Memory"
                      value={fmtBytes(stats.memory_bytes)}
                      hint={`${fmtInt(stats.requests)} requests served`}
                    />
                    {m.task === 'tts' ? (
                      <Readout
                        label="First audio p50"
                        value={fmtMs(stats.latency_ms.ttfa?.p50, false)}
                        unit="ms"
                        hint={`p95 ${fmtMs(stats.latency_ms.ttfa?.p95)}`}
                      />
                    ) : (
                      <>
                        <Readout
                          label="First partial p50"
                          value={fmtMs(stats.latency_ms.first_partial?.p50, false)}
                          unit="ms"
                          hint={`p95 ${fmtMs(stats.latency_ms.first_partial?.p95)}`}
                        />
                        <Readout
                          label="Final p50"
                          value={fmtMs(stats.latency_ms.final?.p50, false)}
                          unit="ms"
                          hint={`p95 ${fmtMs(stats.latency_ms.final?.p95)}`}
                        />
                      </>
                    )}
                  </div>
                ) : (
                  <p className="rounded-lg border border-dashed border-border p-3 text-[13px] leading-5 text-muted-foreground">
                    {m.installed
                      ? 'Load the model and send it some audio; its real-time factor and latency percentiles will appear here.'
                      : isCloud(m)
                        ? 'Its provider is not configured yet. Add it on the Providers screen to use this model.'
                        : 'Not installed. Pull the model to measure it on this machine.'}
                  </p>
                )}
              </section>

              {m.task === 'tts' && !isDemo && (
                <ModelSettingsSection key={`${m.id}:${JSON.stringify(m.settings ?? {})}`} m={m} />
              )}

              {m.measured && Object.keys(m.measured).length > 0 && (
                <section>
                  <h3 className="mb-2 eyebrow">Measured by the maintainers</h3>
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                    {Object.entries(m.measured)
                      .filter(([, v]) => v != null && typeof v !== 'object')
                      .map(([k, v]) => (
                        <div key={k} className="contents">
                          <dt className="text-muted-foreground">{k.replace(/_/g, ' ')}</dt>
                          <dd className="truncate text-right num">
                            {typeof v === 'number' ? (Number.isInteger(v) ? fmtInt(v) : v.toPrecision(3)) : String(v)}
                          </dd>
                        </div>
                      ))}
                  </dl>
                </section>
              )}

              <section>
                <h3 className="mb-2 eyebrow">Specification</h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  <dt className="text-muted-foreground">Engine</dt>
                  <dd className="text-right num">{m.engine}</dd>
                  <dt className="text-muted-foreground">Size on disk</dt>
                  <dd className="text-right num">{fmtBytes(m.size_bytes)}</dd>
                  <dt className="text-muted-foreground">Parameters</dt>
                  <dd className="text-right num">{fmtParams(m.params)}</dd>
                  <dt className="text-muted-foreground">Sample rate</dt>
                  <dd className="text-right num">{fmtInt(m.sample_rate)} Hz</dd>
                  <dt className="text-muted-foreground">Streaming</dt>
                  <dd className="text-right">{m.streaming ? 'Yes' : 'No'}</dd>
                  <dt className="text-muted-foreground">Licence</dt>
                  <dd className="text-right">
                    {m.license_url ? (
                      <a href={m.license_url} target="_blank" rel="noreferrer" className="text-link underline">
                        {m.license}
                      </a>
                    ) : (
                      m.license
                    )}
                  </dd>
                  <dt className="text-muted-foreground">Commercial use</dt>
                  <dd className="text-right">
                    {m.commercial_use == null ? (
                      'not stated'
                    ) : m.commercial_use ? (
                      'Allowed'
                    ) : (
                      <span className="text-warning-text">Not allowed</span>
                    )}
                  </dd>
                  <dt className="text-muted-foreground">Gated</dt>
                  <dd className="text-right">{m.gated ? 'Yes: accept the terms, set HF_TOKEN' : 'No'}</dd>
                  <dt className="text-muted-foreground">Hardware</dt>
                  <dd className="text-right">
                    {m.hardware ? (
                      <>
                        {m.hardware.gpu === 'optional' ? 'GPU optional' : `GPU ${m.hardware.gpu}`}
                        {m.hardware.vram_gb != null && <span className="num">, ~{m.hardware.vram_gb} GB</span>}
                        {m.hardware.note && (
                          <span className="block text-xs text-muted-foreground">{m.hardware.note}</span>
                        )}
                      </>
                    ) : (
                      'Runs on a CPU'
                    )}
                  </dd>
                  {(m.requires?.length ?? 0) > 0 && (
                    <>
                      <dt className="text-muted-foreground">Needs</dt>
                      <dd className="text-right num">{m.requires?.join(', ')}</dd>
                    </>
                  )}
                  {m.status && (
                    <>
                      <dt className="text-muted-foreground">Status</dt>
                      <dd className="text-right">{isPending(m) ? 'Files not published yet' : m.status}</dd>
                    </>
                  )}
                  {(m.phonemizers?.length ?? 0) > 0 && (
                    <>
                      <dt className="text-muted-foreground">Phonemisers</dt>
                      <dd className="text-right">
                        {m.phonemizers
                          ?.map(
                            (p) =>
                              `${PHONEMIZER_LABEL[p.id] ?? p.id}${p.default ? ' (default)' : ''}${p.available ? '' : ' (not installed)'}`,
                          )
                          .join(', ')}
                      </dd>
                    </>
                  )}
                  {m.aliases.length > 0 && (
                    <>
                      <dt className="text-muted-foreground">Aliases</dt>
                      <dd className="text-right num">{m.aliases.join(', ')}</dd>
                    </>
                  )}
                </dl>
              </section>

              {m.gated && (
                <p className="rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] leading-5">
                  This model is gated. Accept its terms on the source page and start the server with a Hugging Face
                  token before pulling.
                </p>
              )}
              {(m.commercial_use === false || !/^(mit|apache|bsd|isc|cc0|unlicense|public)/i.test(m.license)) && (
                <p className="rounded-lg border border-border bg-muted/40 p-3 text-[13px] leading-5 text-muted-foreground">
                  Licence <span className="text-foreground">{m.license}</span>{' '}
                  {m.commercial_use === false
                    ? 'does not allow commercial use.'
                    : 'is not a standard permissive software licence.'}{' '}
                  Check that its terms fit your use.
                </p>
              )}

              <section>
                <h3 className="mb-2 eyebrow">Languages</h3>
                {m.languages.length === 0 ? (
                  <p className="text-[13px] text-muted-foreground">Language independent.</p>
                ) : m.languages.length > 16 ? (
                  <p className="text-[13px] leading-5 text-muted-foreground">
                    <span className="text-foreground">{m.languages.length} languages</span>, including{' '}
                    {m.languages.slice(0, 14).map(languageName).join(', ')}.
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {m.languages.map((l) => (
                      <Badge key={l} variant="secondary" className="font-normal">
                        {languageName(l)}
                      </Badge>
                    ))}
                  </div>
                )}
              </section>

              {m.features.length > 0 && (
                <section>
                  <h3 className="mb-2 eyebrow">Features</h3>
                  <div className="flex flex-wrap gap-1.5">
                    {m.features.map((f) => (
                      <Badge key={f} variant="outline" className="font-normal">
                        {f}
                      </Badge>
                    ))}
                  </div>
                </section>
              )}

              {m.voices.length > 0 && (
                <section>
                  <h3 className="mb-2 eyebrow">Voices ({m.voices.length})</h3>
                  <ul className="grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
                    {m.voices.map((v) => (
                      <li key={v.id} className="flex items-baseline justify-between gap-2">
                        <span className="truncate">{v.name}</span>
                        <span className="shrink-0 num text-xs text-muted-foreground">{v.language}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <div className="flex flex-wrap gap-2 border-t border-border pt-4">
                {(m.task === 'stt' || m.task === 'tts') && m.installed && (
                  <Button
                    size="sm"
                    onClick={() => {
                      setPref(m.task === 'stt' ? 'sttModel' : 'ttsModel', m.id)
                      navigate(m.task === 'stt' ? '/stt' : '/tts')
                    }}
                  >
                    <Play className="fill-current" /> Try in playground
                  </Button>
                )}
                <Button size="sm" variant="outline" asChild>
                  <a href={m.source_url} target="_blank" rel="noreferrer">
                    <ExternalLink /> Source
                  </a>
                </Button>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}

export default function ModelsPage() {
  const catalog = useCatalog()
  const feed = useLive()
  const actions = useModelActions()
  const { modelsView: view, modelsSort: sort, pickDismissed } = useStore(prefs)
  const [q, setQ] = useState('')
  const [task, setTask] = useState<string>('all')
  const [tier, setTier] = useState(ANY)
  const [lang, setLang] = useState(ANY)
  const [license, setLicense] = useState(ANY)
  const [family, setFamily] = useState(ANY)
  const [installedOnly, setInstalledOnly] = useState(false)
  const [noExtras, setNoExtras] = useState(false)
  const [commercialOnly, setCommercialOnly] = useState(false)
  const [cpuOnly, setCpuOnly] = useState(false)
  // Off by default, and never remembered: hardware only reorders and highlights unless this is switched on.
  const [hideUnlikely, setHideUnlikely] = useState(false)
  const [picking, setPicking] = useState(false)
  const [where, setWhere] = useState(ANY)
  const [openId, setOpenId] = useState<string | null>(null)
  const [toDelete, setToDelete] = useState<CatalogModel | null>(null)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const all = useMemo(() => catalog.data?.models ?? [], [catalog.data])
  const statsById = useMemo(() => new Map((feed.stats?.models ?? []).map((m) => [m.id, m])), [feed.stats])
  // The server rates each model for its own machine. The live demo has no server: the machine is the browser's.
  const fitOffered = !isDemo && all.some((m) => m.fit)
  const fitLang = lang === ANY ? null : lang
  const byFit = fitOffered && sort === 'fit'
  const maxMem = Math.max(1, ...[...statsById.values()].map((m) => m.memory_bytes))
  const options = useMemo(() => {
    const uniq = (xs: string[]) => [...new Set(xs)].sort((a, b) => a.localeCompare(b))
    return {
      tasks: uniq(all.map((m) => m.task)) as Task[],
      langs: uniq(all.flatMap((m) => m.languages)),
      licenses: uniq(all.map((m) => m.license)),
      families: uniq(all.map((m) => m.family)),
    }
  }, [all])

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const rank = (m: CatalogModel) =>
      m.loaded
        ? 0
        : m.pulling
          ? 1
          : m.installed
            ? 2
            : isPending(m) || isNotImplemented(m)
              ? 5
              : m.validated === 'cpu'
                ? 3
                : 4
    const base = all
      .filter(
        (m) =>
          (task === 'all' || m.task === task) &&
          (tier === ANY || m.tier === tier) &&
          (lang === ANY || m.languages.includes(lang) || m.languages.includes('multi')) &&
          (license === ANY || m.license === license) &&
          (family === ANY || m.family === family) &&
          (!installedOnly || m.installed) &&
          (!noExtras || !(m.requires?.length ?? 0)) &&
          (!commercialOnly || m.commercial_use !== false) &&
          (!cpuOnly || isCpuFriendly(m)) &&
          (!(fitOffered && hideUnlikely) || !isUnlikely(m)) &&
          (where === ANY || (where === 'cloud') === isCloud(m)) &&
          (!needle ||
            `${m.id} ${m.aliases.join(' ')} ${m.family} ${m.description} ${m.engine}`.toLowerCase().includes(needle)),
      )
      .sort((a, b) => rank(a) - rank(b) || a.family.localeCompare(b.family) || a.id.localeCompare(b.id))
    return byFit ? sortByFit(base, fitLang) : base
  }, [
    all,
    q,
    task,
    tier,
    lang,
    license,
    family,
    installedOnly,
    noExtras,
    commercialOnly,
    cpuOnly,
    where,
    fitOffered,
    hideUnlikely,
    byFit,
    fitLang,
  ])
  const unlikelyHidden = fitOffered && hideUnlikely ? all.filter(isUnlikely).length : 0

  const filtersActive =
    q !== '' ||
    task !== 'all' ||
    tier !== ANY ||
    lang !== ANY ||
    license !== ANY ||
    family !== ANY ||
    installedOnly ||
    noExtras ||
    commercialOnly ||
    cpuOnly ||
    hideUnlikely ||
    where !== ANY
  const reset = () => {
    setWhere(ANY)
    setQ('')
    setTask('all')
    setTier(ANY)
    setLang(ANY)
    setLicense(ANY)
    setFamily(ANY)
    setInstalledOnly(false)
    setNoExtras(false)
    setCommercialOnly(false)
    setCpuOnly(false)
    setHideUnlikely(false)
  }
  const nothingInstalled = !catalog.isLoading && all.length > 0 && all.every((m) => !m.installed || isCloud(m))
  const open = all.find((m) => m.id === openId) ?? null
  const installed = all.filter((m) => m.installed)
  const loaded = all.filter((m) => m.loaded)
  const memTotal = [...statsById.values()].reduce((a, m) => a + m.memory_bytes, 0)
  const serverSide = isDemo ? filtered.filter((m) => m.runtime !== 'browser') : filtered
  const groups: { task: string; label: string; note?: string; models: CatalogModel[] }[] = [
    ...(isDemo
      ? [
          {
            task: 'browser',
            label: 'Runs in your browser',
            note: 'downloaded to this device and run with WebAssembly',
            models: filtered.filter((m) => m.runtime === 'browser'),
          },
        ]
      : []),
    ...TASK_ORDER.map((t) => ({
      task: t,
      label: TASK_LABEL[t],
      note: isDemo ? 'available when you run the server' : undefined,
      models: serverSide.filter((m) => m.task === t),
    })),
  ].filter((g) => g.models.length > 0)

  const filterSelect = (
    label: string,
    value: string,
    set: (v: string) => void,
    items: { value: string; label: string }[],
    width: string,
  ) => (
    <Select value={value} onValueChange={set}>
      <SelectTrigger className={cn('h-8', width)} aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent position="popper" align="start" className="max-h-72">
        <SelectItem value={ANY}>{label}: any</SelectItem>
        {items.map((i) => (
          <SelectItem key={i.value} value={i.value}>
            {i.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const toolbar = (
    <>
      <div className="relative min-w-40 flex-1 sm:max-w-64">
        <Search className="pointer-events-none absolute top-2 left-2.5 size-4 text-muted-foreground" aria-hidden />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search models"
          aria-label="Search models"
          className="h-8 pl-8"
          type="search"
        />
      </div>
      <div className="max-w-full overflow-x-auto">
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={task}
          onValueChange={(v) => v && setTask(v)}
          aria-label="Task"
        >
          <ToggleGroupItem value="all" className="px-2.5 text-xs">
            All
          </ToggleGroupItem>
          {options.tasks.map((t) => (
            <ToggleGroupItem key={t} value={t} className="px-2.5 text-xs" title={TASK_LABEL[t] ?? t}>
              {TASK_SHORT[t] ?? t}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      {filterSelect(
        'Tier',
        tier,
        setTier,
        TIERS.map((t) => ({ value: t, label: TIER_LABEL[t] })),
        'w-32',
      )}
      {filterSelect(
        'Language',
        lang,
        setLang,
        (() => {
          const g = groupLanguages(options.langs)
          return [...g.indic, ...g.english, ...g.other].map((l) => ({ value: l, label: languageLabel(l) }))
        })(),
        'w-36',
      )}
      {filterSelect(
        'Family',
        family,
        setFamily,
        options.families.map((f) => ({ value: f, label: f })),
        'w-36',
      )}
      {filterSelect(
        'Licence',
        license,
        setLicense,
        options.licenses.map((l) => ({ value: l, label: l })),
        'w-36',
      )}
      {all.some((m) => isCloud(m)) &&
        filterSelect(
          'Runs',
          where,
          setWhere,
          [
            { value: 'local', label: 'On this server' },
            { value: 'cloud', label: 'Cloud' },
          ],
          'w-36',
        )}
      <label className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30">
        Installed
        <Switch checked={installedOnly} onCheckedChange={setInstalledOnly} size="sm" />
      </label>
      <label
        className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
        title="Hide models that need an optional component such as eSpeak NG"
      >
        No extra components
        <Switch checked={noExtras} onCheckedChange={setNoExtras} size="sm" />
      </label>
      <label
        className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
        title="Hide models whose licence does not allow commercial use"
      >
        Commercial use
        <Switch
          checked={commercialOnly}
          onCheckedChange={setCommercialOnly}
          size="sm"
          data-testid="filter-commercial"
        />
      </label>
      <label
        className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
        title="Hide models that need or are much faster with a GPU"
      >
        CPU-friendly
        <Switch checked={cpuOnly} onCheckedChange={setCpuOnly} size="sm" data-testid="filter-cpu" />
      </label>
      {fitOffered && (
        <>
          <Select value={sort} onValueChange={(v) => setPref('modelsSort', v as 'default' | 'fit')}>
            <SelectTrigger className="h-8 w-52" aria-label="Sort" data-testid="sort-models">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              <SelectItem value="default">Sort: installed first</SelectItem>
              <SelectItem value="fit">Sort: best for this machine</SelectItem>
            </SelectContent>
          </Select>
          <label
            className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
            title="Optional, off by default: leaves out the models the server expects not to fit or run well on its hardware. They can still be pulled; switch this off to see them"
          >
            Hide unlikely to fit
            <Switch checked={hideUnlikely} onCheckedChange={setHideUnlikely} size="sm" data-testid="filter-unlikely" />
          </label>
        </>
      )}
      {filtersActive && (
        <Button variant="ghost" size="sm" onClick={reset}>
          <X /> Clear
        </Button>
      )}
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={view}
        onValueChange={(v) => v && setPref('modelsView', v as 'cards' | 'table')}
        aria-label="Layout"
        className="ml-auto"
      >
        <ToggleGroupItem value="cards" aria-label="Cards" className="px-2">
          <LayoutGrid />
        </ToggleGroupItem>
        <ToggleGroupItem value="table" aria-label="Table" className="px-2">
          <Rows3 />
        </ToggleGroupItem>
      </ToggleGroup>
    </>
  )

  return (
    <Page title="Models" toolbar={toolbar}>
      {isDemo && (
        <NeedsServer title="This is a catalog snapshot" className="mb-4">
          The first model below really runs in this tab. The rest is a read-only copy of what a bolo server offered when
          this demo was built: pulling, loading and deleting them needs the server.
        </NeedsServer>
      )}
      <div
        className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-muted-foreground"
        aria-live="polite"
      >
        {catalog.isLoading ? (
          <Skeleton className="h-4 w-72" />
        ) : (
          <>
            <span>
              <span className="num text-foreground">{filtered.length}</span> of{' '}
              <span className="num">{all.length}</span> models
            </span>
            <span>
              <span className="num text-foreground">{installed.length}</span> installed ·{' '}
              <span className="num">{fmtBytes(installed.reduce((a, m) => a + m.size_bytes, 0))}</span> on disk
            </span>
            <span>
              <span className="num text-foreground">{loaded.length}</span> loaded ·{' '}
              <span className="num">{memTotal ? fmtBytes(memTotal) : '–'}</span> in memory
            </span>
            {unlikelyHidden > 0 && (
              <span data-testid="unlikely-hidden">
                <span className="num text-foreground">{unlikelyHidden}</span> unlikely to fit this machine hidden by
                your filter ·{' '}
                <button type="button" className="underline underline-offset-2" onClick={() => setHideUnlikely(false)}>
                  show them
                </button>
              </span>
            )}
          </>
        )}
      </div>

      {fitOffered && nothingInstalled && !pickDismissed && (
        <FirstRunPick
          className="mb-4"
          onPick={() => setPicking(true)}
          onDismiss={() => setPref('pickDismissed', true)}
        />
      )}
      {!isDemo && <HardwareCard className="mb-4" onPick={() => setPicking(true)} />}
      <LoadedModelsPanel className="mb-4" />
      {/* The demo's models are downloaded into this browser: the same manager as on the screens that use them. */}
      {isDemo && <DownloadsPanel className="mb-4" />}
      <ComponentsCard className="mb-5" />

      {catalog.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4" aria-busy="true">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="h-64 rounded-xl" />
          ))}
        </div>
      ) : catalog.isError ? (
        <div className="rounded-xl border border-border bg-card">
          <ErrorState
            title="Could not load the catalog"
            error={describeError(catalog.error)}
            onRetry={() => void catalog.refetch()}
          />
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-card">
          <Empty
            icon={Boxes}
            title={all.length === 0 ? 'The catalog is empty' : 'No models match these filters'}
            action={
              filtersActive ? (
                <Button size="sm" variant="outline" onClick={reset}>
                  Clear filters
                </Button>
              ) : undefined
            }
          >
            {all.length === 0 ? 'This server did not report any models.' : 'Try a different search or remove a filter.'}
          </Empty>
        </div>
      ) : view === 'cards' ? (
        <div className="grid gap-6">
          {groups.map((g) => (
            <section key={g.task} aria-labelledby={`group-${g.task}`}>
              <h2 id={`group-${g.task}`} className="mb-2.5 flex items-baseline gap-2 text-sm font-semibold">
                {g.label}
                <span className="num text-xs font-normal text-muted-foreground">{g.models.length}</span>
                {g.note && <span className="text-xs font-normal text-muted-foreground">· {g.note}</span>}
              </h2>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                {(expanded[g.task] ? g.models : g.models.slice(0, GROUP_LIMIT)).map((m) => (
                  <ModelCard
                    key={m.id}
                    m={m}
                    stats={statsById.get(m.id)}
                    maxMem={maxMem}
                    actions={actions}
                    lang={fitLang}
                    onOpen={() => setOpenId(m.id)}
                    onDelete={setToDelete}
                  />
                ))}
              </div>
              {g.models.length > GROUP_LIMIT && (
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={() => setExpanded((e) => ({ ...e, [g.task]: !e[g.task] }))}
                >
                  {expanded[g.task] ? 'Show fewer' : `Show all ${g.models.length}`}
                </Button>
              )}
            </section>
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border bg-card">
          <Table className="min-w-[860px]">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-4">Model</TableHead>
                <TableHead>Task</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead>Languages</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead>Licence</TableHead>
                <TableHead>Checked</TableHead>
                <TableHead className="w-40">Memory</TableHead>
                <TableHead>State</TableHead>
                <TableHead className="pr-4 text-right">
                  <span className="sr-only">Actions</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((m) => {
                const st = statsById.get(m.id)
                return (
                  <TableRow key={m.id} data-testid="model-row">
                    <TableCell className="pl-4">
                      <button
                        type="button"
                        className="rounded-sm text-left num text-[13px] font-medium hover:underline"
                        onClick={() => setOpenId(m.id)}
                      >
                        {m.id}
                      </button>
                      <div className="text-xs text-muted-foreground">{m.family}</div>
                    </TableCell>
                    <TableCell className="text-xs">{TASK_SHORT[m.task] ?? m.task}</TableCell>
                    <TableCell>
                      <TierBadge tier={m.tier} />
                    </TableCell>
                    <TableCell className="max-w-44 truncate text-xs">{languageSummary(m.languages, 2)}</TableCell>
                    <TableCell className="text-right num text-xs">
                      {isCloud(m) ? <span className="text-muted-foreground">cloud</span> : fmtBytes(m.size_bytes)}
                    </TableCell>
                    <TableCell className="text-xs">
                      <span className="flex items-center gap-2">
                        {m.license}
                        {m.commercial_use === false && <span className="text-warning-text">non-commercial</span>}
                        {m.gated && <GatedMark />}
                        {m.license_note && <LicenseNoteMark note={m.license_note} />}
                        <RemoteCodeMark model={m} />
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="flex flex-col items-start gap-0.5">
                        <CloudMark model={m} />
                        <AvailabilityMark model={m} />
                        <ValidatedMark validated={m.validated} />
                        <HardwareMark hardware={m.hardware} />
                        <UnmetMark m={m} />
                        {fitOffered && (
                          <>
                            <FitLine fit={m.fit} className="max-w-56" />
                            <RecommendedBadge m={m} lang={fitLang} />
                          </>
                        )}
                      </span>
                    </TableCell>
                    <TableCell>
                      {m.loaded && st && st.memory_bytes > 0 ? (
                        <MemoryBar bytes={st.memory_bytes} max={maxMem} />
                      ) : (
                        <span className="text-xs text-muted-foreground">–</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <StateDot model={m} />
                    </TableCell>
                    <TableCell className="pr-4">
                      <div className="flex justify-end">
                        <ModelActions m={m} actions={actions} onDelete={setToDelete} compact />
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      )}

      <Details
        m={open}
        stats={open ? statsById.get(open.id) : undefined}
        lang={fitLang}
        onClose={() => setOpenId(null)}
      />
      {!isDemo && <PickForMe open={picking} onOpenChange={setPicking} />}

      <Dialog open={toDelete != null} onOpenChange={(o) => !o && setToDelete(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Delete <span className="num">{toDelete?.id}</span>?
            </DialogTitle>
            <DialogDescription>
              This removes {toDelete ? fmtBytes(toDelete.size_bytes) : ''} from the server's disk
              {toDelete?.loaded ? ' and unloads the model' : ''}. You can pull it again later.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (toDelete) actions.remove.mutate(toDelete.id)
                setToDelete(null)
              }}
            >
              <Trash2 /> Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Page>
  )
}
