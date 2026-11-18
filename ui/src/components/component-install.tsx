import { useQueryClient } from '@tanstack/react-query'
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  CircleDashed,
  Copy,
  ExternalLink,
  PackagePlus,
  Puzzle,
  Search,
  Trash2,
} from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError, useComponents } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { Component } from '@/lib/api/types'
import { fmtBytes } from '@/lib/format'
import {
  defaultFilter,
  filterPacks,
  isLanguagePack,
  isMissing,
  needText,
  packCode,
  packName,
  pageOf,
  readPacksOpen,
  summarize,
  summaryText,
  writePacksOpen,
  type PackFilter,
} from '@/lib/language-packs'
import { cn } from '@/lib/utils'

/** One sentence on what a component is, that it is optional, and how it is licensed. */
export function componentSentence(c: Component): string {
  const what =
    c.id === 'espeak-ng'
      ? 'eSpeak NG is a small open-source program that some voices use to turn text into phonemes'
      : c.id === 'uccharan'
        ? 'uccharan turns English and Hindi text into the phonemes many voices read'
        : c.kind === 'language-pack'
          ? `${c.name} is data that lets voices read text typed in Latin letters: it is written in the language's own script first`
          : c.kind === 'pytorch'
            ? `${c.name} lets the PyTorch models of the bolo-torch plugin use the ${c.gpu ?? 'GPU'} instead of the CPU`
            : c.kind === 'tts-server' || c.id === 'llama-server' || c.id === 'audio.cpp'
              ? `${c.name} is a separate speech server that bolo starts for the voices that run on it`
              : c.kind === 'engine-plugin'
                ? `${c.name} runs models in a process of its own, for models that need PyTorch`
                : `${c.name} is an extra piece of software that some models use`
  const licence = c.license && !/^see /i.test(c.license) ? ` under ${c.license}` : ''
  return `${what}; it is optional, is not part of bolo, and is licensed separately${licence}.`
}

/** Whether installing asks the user to accept a licence first (copyleft and share-alike components). */
export function needsLicenceAcceptance(c: Pick<Component, 'id' | 'license'>): boolean {
  return c.id === 'espeak-ng' || /^(a?gpl|lgpl)/i.test(c.license) || /^cc[ -]by[ -]sa/i.test(c.license)
}

function hostOf(url: string | null | undefined): string | null {
  if (!url) return null
  try {
    return new URL(url).host
  } catch {
    return null
  }
}

/** What the server does when asked to install this component. */
export function installDescription(c: Component): string {
  if (c.kind === 'tts-server' || c.release) {
    const host = hostOf(c.release?.url)
    const version = c.release?.version ? `${c.release.version} ` : ''
    return `The server downloads the prebuilt ${version}release${host ? ` from ${host}` : ''}, checks its SHA-256, unpacks it into its data directory and checks that the program starts on this machine. Nothing else on the machine changes, and the server keeps running.`
  }
  if (c.kind === 'pytorch')
    return `The server stops the bolo-torch plugin, reinstalls torch (and torchaudio and torchvision where present) from PyTorch's own index in each of its environments (${c.size_bytes ? `about ${fmtBytes(c.size_bytes)}` : 'a few GB'} to download), checks that it reaches the GPU, and starts the plugin again on the next request. Models of the plugin are unloaded meanwhile.`
  if (c.kind === 'language-pack')
    return "The server downloads the pack from uccharan's release, checks its SHA-256 and keeps it in uccharan's cache directory. Nothing else on the machine changes, and the server keeps running."
  return "A Python package is installed into the server's own environment. Nothing else on the machine changes, and the server keeps running."
}

const SOURCE_LABEL: Record<string, string> = {
  'system-executable': 'found on the system',
  'python-package': 'Python package',
  configured: 'configured path',
}

/** A command on one line with a Copy button. */
export function CopyCommand({
  command,
  className,
  label = 'Manual install command',
}: {
  command: string
  className?: string
  label?: string
}) {
  const [copied, setCopied] = useState(false)
  if (!command) return null
  return (
    <div
      className={cn(
        'flex min-w-0 items-stretch overflow-hidden rounded-lg border border-border bg-background/60',
        className,
      )}
    >
      <pre
        className="min-w-0 flex-1 overflow-x-auto px-3 py-2 num text-[12.5px] leading-5"
        tabIndex={0}
        aria-label={label}
      >
        <code>{command}</code>
      </pre>
      <Button
        variant="ghost"
        size="sm"
        className="h-auto shrink-0 rounded-none border-l border-border px-3"
        onClick={() => {
          void navigator.clipboard.writeText(command).then(
            () => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1500)
            },
            () => toast.error('Could not copy to the clipboard'),
          )
        }}
      >
        {copied ? <Check /> : <Copy />}
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  )
}

class InstallFailed extends Error {}

/** Starts a server-side install and follows its job until it is done (throws with the job's error). */
export async function installAndWait(id: string, onProgress?: (progress: number) => void): Promise<void> {
  const { job_id } = await api.installComponent(id)
  for (let failures = 0; ;) {
    await new Promise((r) => setTimeout(r, 400))
    try {
      const job = await api.job(job_id)
      failures = 0
      if (job.state === 'error') throw new InstallFailed(job.error ?? 'The install failed.')
      if (job.state === 'done') return
      onProgress?.(job.progress ?? 0)
    } catch (err) {
      if (err instanceof InstallFailed || ++failures >= 5) throw err
    }
  }
}

/** After an install or removal: the component list, the catalog (which models are usable) and the server info. */
function refreshAfterComponentChange(qc: ReturnType<typeof useQueryClient>) {
  return Promise.all([
    qc.invalidateQueries({ queryKey: ['components'] }),
    qc.invalidateQueries({ queryKey: ['catalog'] }),
    qc.invalidateQueries({ queryKey: ['info'] }),
  ])
}

type Phase = { step: 'confirm' } | { step: 'running'; progress: number } | { step: 'error'; message: string }

/**
 * Confirms the licence, starts the server-side install, and follows the job.
 * When it finishes the catalog and component list are refreshed, so the
 * voice that needed the component becomes usable without a reload.
 */
export function InstallComponentDialog({
  component,
  open,
  onOpenChange,
}: {
  component: Component
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const qc = useQueryClient()
  const [phase, setPhase] = useState<Phase>({ step: 'confirm' })
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const install = async () => {
    setPhase({ step: 'running', progress: 0 })
    try {
      await installAndWait(component.id, (progress) => alive.current && setPhase({ step: 'running', progress }))
      await refreshAfterComponentChange(qc)
      toast.success(`${component.name} installed`, { description: 'Models that need it can be used now.' })
      if (alive.current) {
        setPhase({ step: 'confirm' })
        onOpenChange(false)
      }
    } catch (err) {
      if (alive.current) setPhase({ step: 'error', message: describeError(err) })
    }
  }

  const running = phase.step === 'running'
  return (
    <Dialog open={open} onOpenChange={(o) => (!running || o) && onOpenChange(o)}>
      <DialogContent className="sm:max-w-md" data-testid="install-dialog">
        <DialogHeader>
          <DialogTitle>Install {component.name}?</DialogTitle>
          <DialogDescription>{componentSentence(component)}</DialogDescription>
        </DialogHeader>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]">
          <dt className="text-muted-foreground">Licence</dt>
          <dd>
            {component.license_url ? (
              <a
                href={component.license_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-link underline underline-offset-2"
              >
                {component.license} <ExternalLink className="size-3" aria-hidden />
              </a>
            ) : (
              component.license
            )}
          </dd>
          <dt className="text-muted-foreground">What happens</dt>
          <dd>{installDescription(component)}</dd>
        </dl>
        {phase.step === 'running' && (
          <div role="status">
            <Progress
              value={Math.max(4, phase.progress * 100)}
              className="h-1.5"
              aria-label={`Installing ${component.name}`}
            />
            <p className="mt-1.5 text-xs text-muted-foreground">
              Installing… {phase.progress > 0 ? `${Math.round(phase.progress * 100)}%` : ''}
            </p>
          </div>
        )}
        {phase.step === 'error' && (
          <div role="alert" className="grid gap-2 rounded-lg border border-critical/30 bg-critical/10 p-3 text-[13px]">
            <p>
              <span className="font-medium">The install did not complete.</span> {phase.message}
            </p>
            {component.manual_command && (
              <>
                <p className="text-muted-foreground">You can install it yourself instead:</p>
                <CopyCommand command={component.manual_command} />
              </>
            )}
          </div>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={running}>
            {phase.step === 'error' ? 'Close' : 'Cancel'}
          </Button>
          <Button onClick={() => void install()} disabled={running} data-testid="install-confirm">
            <PackagePlus />{' '}
            {running
              ? 'Installing…'
              : phase.step === 'error'
                ? 'Try again'
                : needsLicenceAcceptance(component)
                  ? 'Accept the licence and install'
                  : 'Install'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Inline (never modal) notice for a model that cannot run until an optional
 * component is installed: what the component is, and either an install button
 * or, when the server cannot install it, the reason and the manual command.
 */
export function ComponentNotice({
  component,
  subject,
  className,
}: {
  component: Component
  subject: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div
      className={cn('rounded-xl border border-warning/50 bg-warning/10 p-4', className)}
      role="note"
      data-testid="component-notice"
    >
      <div className="flex items-start gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground">
          <Puzzle className="size-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold">
            {subject} needs {component.name}
          </h2>
          <p className="mt-1 max-w-2xl text-[13px] leading-5 text-muted-foreground">{componentSentence(component)}</p>
          {isDemo ? (
            <p className="mt-2 text-[13px] leading-5">
              Installing components is a server feature; this demo runs in your browser only.
            </p>
          ) : component.installable ? (
            <Button size="sm" className="mt-3" onClick={() => setOpen(true)} data-testid="install-component">
              <PackagePlus /> Install {component.name}
            </Button>
          ) : (
            <div className="mt-3 grid max-w-2xl gap-2">
              <p className="text-[13px] leading-5">
                <span className="font-medium">The server cannot install it for you.</span>{' '}
                {component.install_blocked_reason ?? 'Install it yourself, then it is picked up without a restart.'}
              </p>
              <CopyCommand command={component.manual_command} />
            </div>
          )}
        </div>
      </div>
      {!isDemo && component.installable && (
        <InstallComponentDialog component={component} open={open} onOpenChange={setOpen} />
      )}
    </div>
  )
}

/** Small mark for cards and rows. */
export function NeedsComponentMark({ name, className }: { name: string; className?: string }) {
  return (
    <span
      className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}
      title={`Cannot be used until ${name} is installed`}
    >
      <Puzzle className="size-3.5" aria-hidden /> Needs {name}
    </span>
  )
}

/** Install button with its dialog, for places that name a missing component (model cards, notices). */
export function ComponentInstallButton({
  component,
  label,
  className,
  size = 'sm',
}: {
  component: Component
  label?: string
  className?: string
  size?: 'sm' | 'xs'
}) {
  const [open, setOpen] = useState(false)
  if (isDemo || component.installed || !component.installable) return null
  return (
    <>
      <Button
        size={size}
        variant="outline"
        className={className}
        onClick={() => setOpen(true)}
        data-testid={`install-${component.id}`}
      >
        <PackagePlus /> {label ?? `Install ${component.name}`}
      </Button>
      <InstallComponentDialog component={component} open={open} onOpenChange={setOpen} />
    </>
  )
}

export const KIND_LABEL: Record<string, string> = {
  phonemizer: 'Phonemisers',
  'language-pack': 'Language packs',
  'tts-server': 'Speech servers',
  'engine-plugin': 'Engine plugins',
  pytorch: 'PyTorch',
  other: 'Other components',
}

export function componentKind(c: Component): string {
  if (c.kind) return c.kind
  if (c.id === 'espeak-ng' || c.id === 'uccharan') return 'phonemizer'
  if (c.id === 'llama-server' || c.id === 'audio.cpp') return 'tts-server'
  if (c.id.startsWith('plugin:')) return 'engine-plugin'
  return 'other'
}

function ComponentRow({ c, detailed }: { c: Component; detailed?: boolean }) {
  const detail: [string, string][] = detailed
    ? ([
        c.path ? ['Program', c.path] : null,
        c.servers?.length ? ['Connected servers', c.servers.join(', ')] : null,
        !c.installed && c.release ? ['Would install', `${c.release.version} (prebuilt, checksum pinned)`] : null,
        c.languages?.length ? ['Languages', c.languages.join(', ')] : null,
        c.torch_compile
          ? [
              'Speed-up (torch.compile)',
              c.torch_compile.speed_up
                ? `on${c.torch_compile.triton_version ? ` (Triton ${c.torch_compile.triton_version})` : ''}`
                : `off: ${c.torch_compile.reason ?? 'unknown'}`,
            ]
          : null,
        ...(c.envs ?? []).map((e): [string, string] => [
          `PyTorch (${e.plugins.join(', ')})`,
          `${e.torch ?? 'not installed'}: ${e.label}`,
        ]),
      ].filter(Boolean) as [string, string][])
    : []
  return (
    <li
      className="grid gap-x-4 gap-y-1 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center"
      data-testid={`component-${c.id}`}
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
          <span className="text-[13px] font-medium">{c.name}</span>
          <span
            className={cn(
              'inline-flex items-center gap-1 text-xs',
              c.installed ? 'text-good-text' : 'text-muted-foreground',
            )}
          >
            {c.installed ? (
              <CircleCheck className="size-3.5" aria-hidden />
            ) : (
              <CircleDashed className="size-3.5" aria-hidden />
            )}
            {c.installed ? 'Installed' : c.servers?.length ? 'Connected' : 'Not installed'}
          </span>
        </div>
        <div className="mt-0.5 text-xs leading-5 text-muted-foreground">
          {c.installed && (
            <>
              {c.source ? (SOURCE_LABEL[c.source] ?? c.source) : 'source unknown'}
              {c.version ? <span className="num"> · {c.version}</span> : null}
              {' · '}
            </>
          )}
          {c.license_url ? (
            <a href={c.license_url} target="_blank" rel="noreferrer" className="text-link underline underline-offset-2">
              {c.license}
            </a>
          ) : (
            c.license
          )}
          {' · '}
          needed by <span className="num">{c.required_by_count}</span> model{c.required_by_count === 1 ? '' : 's'}
        </div>
        {detailed && c.description && (
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">{c.description}</p>
        )}
        {detail.length > 0 && (
          <dl className="mt-1.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-xs">
            {detail.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="min-w-0 num break-all">{v}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
      {!c.installed && (
        <ComponentInstallButton component={c} label={c.kind === 'pytorch' ? 'Install GPU version' : 'Install'} />
      )}
      {!c.installed && !isDemo && !c.installable && (
        <div className="min-w-0 sm:col-span-2">
          <p className="mb-1.5 text-xs leading-5 text-muted-foreground">
            {c.install_blocked_reason ?? 'The server cannot install this; do it yourself:'}
          </p>
          <CopyCommand command={c.manual_command} />
        </div>
      )}
    </li>
  )
}

function PackRow({ c }: { c: Component }) {
  const qc = useQueryClient()
  const [removing, setRemoving] = useState(false)
  const name = packName(c)
  const missing = isMissing(c)
  const Icon = c.installed ? CircleCheck : missing ? CircleAlert : CircleDashed
  const status = c.installed ? 'Installed' : missing ? 'Needed, not installed' : 'Not installed'
  const remove = async () => {
    setRemoving(true)
    try {
      await api.removeComponent(c.id)
      await refreshAfterComponentChange(qc)
      toast.success(`${name} pack removed`)
    } catch (err) {
      toast.error(`Could not remove the ${name} pack`, { description: describeError(err) })
    } finally {
      setRemoving(false)
    }
  }
  const meta = [
    needText(c),
    c.size_bytes ? fmtBytes(c.size_bytes) : null,
    /^cc[ -]by[ -]sa/i.test(c.license) ? c.license : null,
  ]
    .filter(Boolean)
    .join(' · ')
  return (
    <li className="flex min-h-10 items-center gap-2.5 py-1.5" data-testid={`component-${c.id}`}>
      <Icon
        className={cn(
          'size-4 shrink-0',
          c.installed ? 'text-good-text' : missing ? 'text-warning-text' : 'text-muted-foreground',
        )}
        aria-label={status}
        role="img"
      />
      <div className="min-w-0 flex-1 @lg/packs:flex @lg/packs:items-baseline @lg/packs:gap-3">
        <div className="flex min-w-0 items-baseline gap-1.5 @lg/packs:w-64 @lg/packs:shrink-0">
          <span className="truncate text-[13px] font-medium" title={c.name}>
            {name}
          </span>
          <span className="shrink-0 num text-[11px] text-muted-foreground">{packCode(c)}</span>
        </div>
        <div className="truncate text-xs leading-4 text-muted-foreground" title={meta}>
          {meta}
        </div>
      </div>
      {c.installed ? (
        c.removable && !isDemo ? (
          <Button
            size="xs"
            variant="ghost"
            className="shrink-0 text-muted-foreground"
            onClick={() => void remove()}
            disabled={removing}
            aria-label={`Remove the ${name} pack`}
            data-testid={`remove-${c.id}`}
          >
            <Trash2 /> {removing ? 'Removing…' : 'Remove'}
          </Button>
        ) : null
      ) : c.installable ? (
        <ComponentInstallButton component={c} label="Install" size="xs" className="shrink-0" />
      ) : !isDemo && !c.built_in ? (
        <span className="shrink-0 text-xs text-muted-foreground" title={c.install_blocked_reason ?? c.manual_command}>
          Manual
        </span>
      ) : null}
    </li>
  )
}

const FILTER_LABEL: Record<PackFilter, string> = { needed: 'Needed', installed: 'Installed', all: 'All' }

/** The packs as a compact list: search, Needed / Installed / All, pages of eight, and "Install needed". */
function PackList({ packs }: { packs: Component[] }) {
  const qc = useQueryClient()
  const [filter, setFilter] = useState<PackFilter>(() => defaultFilter(packs))
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)
  const [bulk, setBulk] = useState<{ done: number; total: number } | null>(null)
  const shown = filterPacks(packs, filter, query)
  const p = pageOf(shown, page)
  const summary = summarize(packs)
  const toInstall = summary.mine ? packs.filter((c) => isMissing(c) && c.installable && !needsLicenceAcceptance(c)) : []
  const installNeeded = async () => {
    setBulk({ done: 0, total: toInstall.length })
    let failed = 0
    for (const [i, c] of toInstall.entries()) {
      try {
        await installAndWait(c.id)
      } catch (err) {
        failed++
        toast.error(`Could not install the ${packName(c)} pack`, { description: describeError(err) })
      }
      setBulk({ done: i + 1, total: toInstall.length })
    }
    await refreshAfterComponentChange(qc)
    setBulk(null)
    if (failed < toInstall.length) {
      const n = toInstall.length - failed
      toast.success(`${n} language pack${n === 1 ? '' : 's'} installed`)
    }
  }
  const counts: Record<PackFilter, number> = {
    needed: summary.needed,
    installed: summary.installed,
    all: summary.total,
  }
  return (
    <div className="@container/packs mt-2 grid grid-cols-[minmax(0,1fr)] gap-2" data-testid="language-pack-list">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 basis-40">
          <Search
            className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground"
            aria-hidden
          />
          <Input
            type="search"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setPage(0)
            }}
            placeholder="Search packs"
            aria-label="Search language packs"
            className="h-7 pl-7 text-xs md:text-xs"
          />
        </div>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={filter}
          onValueChange={(v) => {
            if (v) {
              setFilter(v as PackFilter)
              setPage(0)
            }
          }}
          aria-label="Show"
        >
          {(['needed', 'installed', 'all'] as const).map((f) => (
            <ToggleGroupItem key={f} value={f} className="px-2 text-xs" data-testid={`packs-filter-${f}`}>
              {FILTER_LABEL[f]} <span className="num text-muted-foreground">{counts[f]}</span>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      {toInstall.length > 0 && !isDemo && (
        <Button
          size="sm"
          className="justify-self-start"
          onClick={() => void installNeeded()}
          disabled={bulk !== null}
          data-testid="install-needed-packs"
        >
          <PackagePlus />
          {bulk
            ? `Installing ${Math.min(bulk.done + 1, bulk.total)} of ${bulk.total}…`
            : `Install needed (${toInstall.length})`}
        </Button>
      )}
      {p.items.length ? (
        <ul className="divide-y divide-border" aria-label="Language packs">
          {p.items.map((c) => (
            <PackRow key={c.id} c={c} />
          ))}
        </ul>
      ) : (
        <p className="py-3 text-xs text-muted-foreground">
          {query
            ? `No pack matches “${query}”.`
            : filter === 'needed'
              ? 'No model installed here needs a pack.'
              : 'None.'}
        </p>
      )}
      {p.total > 0 && (
        <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="num" aria-live="polite" data-testid="packs-range">
            {p.from}–{p.to} of {p.total}
          </span>
          {p.pages > 1 && (
            <span className="flex gap-1">
              <Button
                size="icon-xs"
                variant="outline"
                onClick={() => setPage(p.page - 1)}
                disabled={p.page === 0}
                aria-label="Previous page"
              >
                <ChevronLeft />
              </Button>
              <Button
                size="icon-xs"
                variant="outline"
                onClick={() => setPage(p.page + 1)}
                disabled={p.page >= p.pages - 1}
                aria-label="Next page"
              >
                <ChevronRight />
              </Button>
            </span>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * uccharan's language packs as one row: how many are installed and needed, with the list behind View (closed by
 * default, remembered in this browser). With `collapsible` false the list is always shown (Components page).
 */
export function LanguagePacks({ packs, collapsible = true }: { packs: Component[]; collapsible?: boolean }) {
  const [open, setOpen] = useState(() => !collapsible || readPacksOpen())
  const listId = useId()
  const summary = summarize(packs)
  const toggle = () => {
    setOpen(!open)
    writePacksOpen(!open)
  }
  return (
    <li className="py-2.5" data-testid="language-packs">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          {collapsible && <span className="text-[13px] font-medium">Language packs</span>}
          <p
            className={cn('text-xs leading-5', summary.missing ? 'text-warning-text' : 'text-muted-foreground')}
            data-testid="language-packs-summary"
          >
            {summaryText(summary)}
          </p>
        </div>
        {collapsible && (
          <Button
            size="sm"
            variant="outline"
            className="shrink-0"
            onClick={toggle}
            aria-expanded={open}
            aria-controls={listId}
            data-testid="language-packs-toggle"
          >
            {open ? 'Hide' : 'View'}
            <ChevronDown className={cn('transition-transform', open && 'rotate-180')} aria-hidden />
          </Button>
        )}
      </div>
      {open && (
        <div id={listId}>
          <PackList packs={packs} />
        </div>
      )}
    </li>
  )
}

/** Compact list of optional components: status, where it came from, licence, and who needs it. */
export function ComponentsCard({
  className,
  bare,
  detailed,
  kind,
}: {
  className?: string
  bare?: boolean
  /** Also show description, program path, connected servers and the release that would be installed. */
  detailed?: boolean
  /** Only components of this kind. */
  kind?: string
}) {
  const q = useComponents()
  const list = (q.data?.components ?? []).filter((c) => !kind || componentKind(c) === kind)
  if (!q.isLoading && !q.isError && list.length === 0) return null
  const packs = list.filter(isLanguagePack)
  // the packs (35 or more) are one row, where the first of them was
  const firstPack = list.findIndex(isLanguagePack)
  const rows = list.filter((c) => !isLanguagePack(c))
  const body = q.isLoading ? (
    <Skeleton className="h-10 w-full" />
  ) : q.isError ? (
    <p className="text-xs text-muted-foreground">Component status is unavailable.</p>
  ) : (
    <>
      <ul className="divide-y divide-border">
        {rows.slice(0, firstPack < 0 ? rows.length : firstPack).map((c) => (
          <ComponentRow key={c.id} c={c} detailed={detailed} />
        ))}
        {packs.length > 0 && <LanguagePacks packs={packs} collapsible={kind !== 'language-pack'} />}
        {firstPack >= 0 && rows.slice(firstPack).map((c) => <ComponentRow key={c.id} c={c} detailed={detailed} />)}
      </ul>
      {isDemo && (
        <p className="mt-1 text-xs leading-5 text-muted-foreground">
          Read-only here: installing components is a server feature.
        </p>
      )}
    </>
  )
  if (bare) return <div className={className}>{body}</div>
  return (
    <Panel
      title="Optional components"
      description="Not shipped with bolo; some models need them"
      className={className}
      bodyClassName="py-1.5"
      data-testid="components-card"
    >
      {body}
    </Panel>
  )
}
