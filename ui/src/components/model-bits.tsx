import {
  Ban,
  CircleCheck,
  CircleDashed,
  Clock,
  Cloud,
  Code,
  Cpu,
  FileExclamationPoint,
  Lock,
  PackageX,
  Scale,
  Target,
  Zap,
} from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { CatalogModel, Tier } from '@/lib/api/types'
import {
  cloudNotice,
  hardwareLabel,
  isCloud,
  isNotImplemented,
  isPending,
  notRunnableReason,
  remoteCodeCommand,
} from '@/lib/capabilities'
import { languageName } from '@/lib/format'
import { cn } from '@/lib/utils'

export const TIER_LABEL: Record<Tier, string> = { realtime: 'Realtime', balanced: 'Balanced', accuracy: 'Accuracy' }
export const TIER_HINT: Record<Tier, string> = {
  realtime: 'Built for live use: first words in well under a second on CPU.',
  balanced: 'A middle ground between speed and accuracy.',
  accuracy: 'Most accurate; usually too slow for live use without a GPU.',
}
export const TIERS: Tier[] = ['realtime', 'balanced', 'accuracy']

const TIER_ICON = { realtime: Zap, balanced: Scale, accuracy: Target } as const

export function TierBadge({ tier, className }: { tier: Tier; className?: string }) {
  const Icon = TIER_ICON[tier]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex h-5 items-center gap-1 rounded-md border border-border bg-muted/40 px-1.5 text-[11px] font-medium text-foreground',
            className,
          )}
        >
          <Icon className="size-3 text-muted-foreground" aria-hidden />
          {TIER_LABEL[tier]}
        </span>
      </TooltipTrigger>
      <TooltipContent>{TIER_HINT[tier]}</TooltipContent>
    </Tooltip>
  )
}

export function ValidatedMark({ validated, className }: { validated: CatalogModel['validated']; className?: string }) {
  const ok = validated === 'cpu'
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex items-center gap-1 text-xs',
            ok ? 'text-good-text' : 'text-muted-foreground',
            className,
          )}
        >
          {ok ? <CircleCheck className="size-3.5" aria-hidden /> : <CircleDashed className="size-3.5" aria-hidden />}
          {ok ? 'Validated on CPU' : 'Untested'}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-60">
        {ok
          ? 'The maintainers downloaded and ran this model on CPU.'
          : 'Listed in the catalog but not yet run by the maintainers. It may work; nothing is promised.'}
      </TooltipContent>
    </Tooltip>
  )
}

export function GatedMark({ className }: { className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}>
          <Lock className="size-3.5" aria-hidden /> Gated
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-60">
        You must accept the model's terms on its source page and provide a token before it can be pulled.
      </TooltipContent>
    </Tooltip>
  )
}

export function languageSummary(codes: string[], max = 3): string {
  if (codes.length === 0) return 'Any'
  if (codes.includes('multi')) return 'Multilingual'
  if (codes.length > max + 2) return `${codes.length} languages`
  const names = codes.slice(0, max).map(languageName)
  return codes.length > max ? `${names.join(', ')} +${codes.length - max}` : names.join(', ')
}

export function StateDot({
  model,
  className,
}: {
  model: Pick<CatalogModel, 'loaded' | 'installed' | 'pulling'>
  className?: string
}) {
  const label = model.loaded
    ? 'Loaded'
    : model.pulling
      ? 'Downloading'
      : model.installed
        ? 'Installed'
        : 'Not installed'
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-xs text-muted-foreground', className)}>
      <span
        aria-hidden
        className={cn(
          'size-2 rounded-full',
          model.loaded
            ? 'bg-good'
            : model.pulling
              ? 'live-dot bg-chart-1'
              : model.installed
                ? 'border-[1.5px] border-good'
                : 'border-[1.5px] border-chart-mute',
        )}
      />
      {label}
    </span>
  )
}

/** GPU advice ("GPU recommended, ~2 GB"); nothing for models that are fine on a CPU. */
export function HardwareMark({ hardware, className }: { hardware: CatalogModel['hardware']; className?: string }) {
  const label = hardwareLabel(hardware)
  if (!hardware || !label) return null
  const strong = hardware.gpu === 'required' || hardware.gpu === 'recommended'
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex items-center gap-1 text-xs',
            strong ? 'text-warning-text' : 'text-muted-foreground',
            className,
          )}
          data-testid="hardware-mark"
        >
          <Cpu className="size-3.5" aria-hidden /> {label}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {hardware.gpu === 'optional'
          ? 'Runs on a CPU; a GPU makes it faster.'
          : hardware.gpu === 'required'
            ? 'Too slow to be useful without a GPU.'
            : 'Works on a CPU but far slower than real time; use a GPU for live use.'}
        {hardware.vram_gb != null && ` Needs about ${hardware.vram_gb} GB of GPU memory.`}
        {hardware.note ? ` ${hardware.note.charAt(0).toUpperCase()}${hardware.note.slice(1)}.` : ''}
      </TooltipContent>
    </Tooltip>
  )
}

/** "Cloud" for models behind a paid cloud API, with the notice that audio leaves the server. */
export function CloudMark({
  model,
  className,
}: {
  model: Pick<CatalogModel, 'cloud' | 'engine' | 'notice' | 'provider'>
  className?: string
}) {
  if (!isCloud(model)) return null
  const provider = model.provider?.name
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex h-5 items-center gap-1 rounded-md border border-warning/40 bg-warning/10 px-1.5 text-[11px] font-medium text-warning-text',
            className,
          )}
          data-testid="cloud-mark"
        >
          <Cloud className="size-3" aria-hidden />
          Cloud
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {cloudNotice(model)}.{' '}
        {provider
          ? model.provider?.configured
            ? `Served by the provider ${provider}.`
            : `Needs the provider ${provider}, which is not configured yet.`
          : ''}
      </TooltipContent>
    </Tooltip>
  )
}

/** "Not published yet" for pending entries, "Not runnable yet" for entries no runtime can run. */
export function AvailabilityMark({
  model,
  className,
}: {
  model: Pick<CatalogModel, 'status' | 'description' | 'runnable' | 'not_runnable_reason'>
  className?: string
}) {
  const pending = isPending(model)
  const notImpl = !pending && isNotImplemented(model)
  const why = notRunnableReason(model)
  if (!pending && !notImpl) return null
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}
          data-testid={pending ? 'pending-mark' : 'not-implemented-mark'}
        >
          {pending ? <Clock className="size-3.5" aria-hidden /> : <Ban className="size-3.5" aria-hidden />}
          {pending ? 'Not published yet' : 'Not runnable yet'}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {pending
          ? 'Described in the catalog, but its files are not published yet. It cannot be pulled or used.'
          : `Listed for reference: nothing this server has can run it yet, so loading it fails. ${why ?? 'The description says why.'}`}
      </TooltipContent>
    </Tooltip>
  )
}

/** "Non-commercial" for models whose licence rules out commercial use. */
export function NonCommercialMark({ license, className }: { license: string; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}
          data-testid="non-commercial-mark"
        >
          <Ban className="size-3.5" aria-hidden /> Non-commercial
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        Its licence ({license}) does not allow commercial use. Read it before you build on this model.
      </TooltipContent>
    </Tooltip>
  )
}

/** "Licence note": a term to act on before use, such as a registration for commercial use. */
export function LicenseNoteMark({ note, className }: { note: string; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}
          data-testid="license-note-mark"
        >
          <FileExclamationPoint className="size-3.5" aria-hidden /> Licence note
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">{note}</TooltipContent>
    </Tooltip>
  )
}

/**
 * "Remote code": loading the model runs Python files from its own repository, pinned by commit and sha256, and
 * only after the operator opts in for it. Says which way the opt-in stands.
 */
export function RemoteCodeMark({
  model,
  className,
}: {
  model: Pick<CatalogModel, 'id' | 'remote_code'>
  className?: string
}) {
  const rc = model.remote_code
  if (!rc) return null
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            'inline-flex items-center gap-1 text-xs',
            rc.allowed ? 'text-muted-foreground' : 'text-warning-text',
            className,
          )}
          data-testid="remote-code-mark"
          data-allowed={rc.allowed}
        >
          <Code className="size-3.5" aria-hidden /> {rc.allowed ? 'Remote code allowed' : 'Remote code, not allowed'}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">
        Loading it runs Python files from the model&apos;s own repository with the server&apos;s rights.{' '}
        {rc.allowed
          ? 'You opted in for this model (allow_remote_code).'
          : `It will not load until you opt in: ${remoteCodeCommand(model.id)}.`}
      </TooltipContent>
    </Tooltip>
  )
}

/** "Engine not installed": the server has nothing to run this model with yet (whisper.cpp without pywhispercpp...). */
export function EngineMissingMark({
  engine,
  reason,
  className,
}: {
  engine: string
  reason?: string | null
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn('inline-flex items-center gap-1 text-xs text-warning-text', className)}
          data-testid="engine-missing-mark"
        >
          <PackageX className="size-3.5" aria-hidden /> Needs {engine}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">
        The {engine} engine is not installed on this server, so the model cannot load yet. {reason ?? ''}
      </TooltipContent>
    </Tooltip>
  )
}
