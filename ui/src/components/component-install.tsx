import { useQueryClient } from '@tanstack/react-query'
import { Check, CircleCheck, CircleDashed, Copy, ExternalLink, PackagePlus, Puzzle } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
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
  if (c.kind === 'language-pack')
    return "The server downloads the pack from uccharan's release, checks its SHA-256 and keeps it in uccharan's cache directory. Nothing else on the machine changes, and the server keeps running."
  return "A Python package is installed into the server's own environment. Nothing else on the machine changes, and the server keeps running."
}

const SOURCE_LABEL: Record<string, string> = {
  'system-executable': 'found on the system',
  'python-package': 'Python package',
  configured: 'configured path',
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false)
  if (!command) return null
  return (
    <div className="flex min-w-0 items-stretch overflow-hidden rounded-lg border border-border bg-background/60">
      <pre
        className="min-w-0 flex-1 overflow-x-auto px-3 py-2 num text-[12.5px] leading-5"
        tabIndex={0}
        aria-label="Manual install command"
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
      const { job_id } = await api.installComponent(component.id)
      for (let failures = 0; ;) {
        await new Promise((r) => setTimeout(r, 400))
        try {
          const job = await api.job(job_id)
          failures = 0
          if (job.state === 'error') throw new InstallFailed(job.error ?? 'The install failed.')
          if (job.state === 'done') break
          if (alive.current) setPhase({ step: 'running', progress: job.progress ?? 0 })
        } catch (err) {
          if (err instanceof InstallFailed || ++failures >= 5) throw err
        }
      }
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['components'] }),
        qc.invalidateQueries({ queryKey: ['catalog'] }),
        qc.invalidateQueries({ queryKey: ['info'] }),
      ])
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

class InstallFailed extends Error {}

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
}: {
  component: Component
  label?: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  if (isDemo || component.installed || !component.installable) return null
  return (
    <>
      <Button
        size="sm"
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
      {!c.installed && <ComponentInstallButton component={c} label="Install" />}
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
  const body = q.isLoading ? (
    <Skeleton className="h-10 w-full" />
  ) : q.isError ? (
    <p className="text-xs text-muted-foreground">Component status is unavailable.</p>
  ) : (
    <>
      <ul className="divide-y divide-border">
        {list.map((c) => (
          <ComponentRow key={c.id} c={c} detailed={detailed} />
        ))}
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
