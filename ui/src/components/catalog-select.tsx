import type { ReactNode } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { CatalogModel } from '@/lib/api/types'
import { isCloud, licenceTags, runtimeLabel } from '@/lib/capabilities'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

export const SERVER_DEFAULT = '__default'

/** What a model in a picker still needs before it can answer, in a few words; null when it can. */
export function readiness(m: CatalogModel, autoPull: boolean | null): string | null {
  if (m.fit?.engine_available === false) return `${m.engine} not installed`
  if (m.available === false) return 'needs a component'
  if (m.remote_code && !m.remote_code.allowed) return 'remote code not allowed'
  if (m.installed || m.engine === 'builtin' || isCloud(m) || m.size_bytes === 0) return null
  return autoPull === false ? 'not installed' : 'downloaded on first use'
}

/**
 * Every catalog model of one kind as a select, installed or not, each with its licence, its labels (gated,
 * non-commercial, licence note, remote code) and what it still needs. The first item is the server's default.
 */
export function CatalogSelect({
  models,
  value,
  onChange,
  label,
  defaultLabel,
  disabled,
  autoPull = null,
  extra,
  className,
  testId,
}: {
  models: CatalogModel[]
  /** A model id, or '' for the server's default. */
  value: string
  onChange: (id: string) => void
  label: string
  /** The first item; null leaves it out. */
  defaultLabel: string | null
  disabled?: boolean
  autoPull?: boolean | null
  /** More words for an item (e.g. "reads the transcript"). */
  extra?: (m: CatalogModel) => string | null
  className?: string
  testId?: string
}) {
  const sorted = [...models].sort(
    (a, b) => Number(b.installed) - Number(a.installed) || Number(isCloud(a)) - Number(isCloud(b)),
  )
  const current = models.find((m) => m.id === value)
  return (
    <Select
      value={value || (defaultLabel != null ? SERVER_DEFAULT : undefined)}
      onValueChange={(v) => onChange(v === SERVER_DEFAULT ? '' : v)}
      disabled={disabled}
    >
      <SelectTrigger className={cn('h-8 w-60 min-w-0', className)} aria-label={label} data-testid={testId}>
        <SelectValue placeholder={label}>
          {current ? <span className="num">{current.id}</span> : (defaultLabel ?? undefined)}
        </SelectValue>
      </SelectTrigger>
      <SelectContent
        position="popper"
        align="start"
        className="max-h-[min(26rem,var(--radix-select-content-available-height))] w-[min(30rem,calc(100vw-2rem))]"
      >
        {defaultLabel != null && <SelectItem value={SERVER_DEFAULT}>{defaultLabel}</SelectItem>}
        {sorted.map((m) => {
          const words: ReactNode[] = []
          const more = extra?.(m)
          if (more) words.push(more)
          if (!isCloud(m) && m.size_bytes > 0) words.push(fmtBytes(m.size_bytes))
          const runtime = runtimeLabel(m)
          if (runtime) words.push(runtime)
          words.push(m.license)
          const tags = licenceTags(m)
          const ready = readiness(m, autoPull)
          return (
            <SelectItem key={m.id} value={m.id} textValue={m.id} data-testid={testId ? `${testId}-${m.id}` : undefined}>
              <span className="grid min-w-0 py-0.5">
                <span className="flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className={cn(
                      'size-1.5 shrink-0 rounded-full',
                      m.loaded ? 'bg-good' : m.installed ? 'bg-chart-mute' : 'border border-chart-mute',
                    )}
                  />
                  <span className="num text-[13px]">{m.id}</span>
                </span>
                <span className="text-xs leading-4 whitespace-normal text-muted-foreground">
                  {words.join(' · ')}
                  {tags.length > 0 && <span className="text-warning-text"> · {tags.join(' · ')}</span>}
                  {ready && <span> · {ready}</span>}
                </span>
              </span>
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}
