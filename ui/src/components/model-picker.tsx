import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import type { CatalogModel } from '@/lib/api/types'
import { fmtBytes, languageName } from '@/lib/format'
import { supportsLanguage } from '@/lib/languages'
import { cn } from '@/lib/utils'
import { languageSummary, TIER_LABEL, TIERS } from './model-bits'

/** Model select grouped by tier, showing size, languages and whether it is loaded. */
export function ModelPicker({
  models,
  value,
  onChange,
  loading,
  disabled,
  id,
  className,
  label = 'Model',
  language,
}: {
  models: CatalogModel[]
  value: string | null
  onChange: (id: string) => void
  loading?: boolean
  disabled?: boolean
  id?: string
  className?: string
  label?: string
  /** When set (and not `auto`), only models that cover this language are offered. */
  language?: string
}) {
  if (loading) return <Skeleton className={cn('h-8 w-56', className)} />
  // Catalogs run to hundreds of entries; the picker offers what can be used now.
  const installed = models.filter((m) => m.installed)
  const usable = installed.length ? installed : models
  const filtering = Boolean(language && language !== 'auto')
  const matching = filtering ? usable.filter((m) => supportsLanguage(m.languages, language as string)) : usable
  // Never hide the current choice, and never show an empty list.
  const shown = matching.length ? usable.filter((m) => matching.includes(m) || m.id === value) : usable
  const hidden = models.length - usable.length
  const filteredOut = usable.length - shown.length
  return (
    <Select value={value ?? undefined} onValueChange={onChange} disabled={disabled || models.length === 0}>
      <SelectTrigger
        id={id}
        aria-label={label}
        className={cn('h-8 w-56 min-w-0', className)}
        data-testid="model-picker"
      >
        <SelectValue placeholder={models.length ? 'Choose a model' : 'No models'} />
      </SelectTrigger>
      <SelectContent
        position="popper"
        align="start"
        className="max-h-[min(28rem,var(--radix-select-content-available-height))] min-w-80"
      >
        {TIERS.map((tier) => {
          const group = shown.filter((m) => m.tier === tier)
          if (group.length === 0) return null
          return (
            <SelectGroup key={tier}>
              <SelectLabel className="eyebrow">{TIER_LABEL[tier]}</SelectLabel>
              {group.map((m) => (
                <SelectItem key={m.id} value={m.id} textValue={m.id} disabled={!m.installed} className="py-1.5">
                  <span className="flex w-full min-w-0 items-center gap-2">
                    <span
                      aria-hidden
                      className={cn(
                        'size-1.5 shrink-0 rounded-full',
                        m.loaded ? 'bg-good' : m.installed ? 'bg-chart-mute' : 'border border-chart-mute',
                      )}
                    />
                    <span className="truncate num text-[13px]">{m.id}</span>
                    <span className="ml-auto shrink-0 pl-3 text-xs text-muted-foreground [[data-slot=select-trigger]_&]:hidden">
                      {languageSummary(m.languages, 2)} · {fmtBytes(m.size_bytes)}
                      {m.available === false
                        ? ' · needs a component'
                        : m.loaded
                          ? ' · loaded'
                          : m.installed
                            ? ''
                            : ' · not installed'}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectGroup>
          )
        })}
        {filteredOut > 0 && (
          <p
            className="border-t border-border px-3 py-2 text-xs text-muted-foreground"
            data-testid="picker-language-note"
          >
            Showing models that support {languageName(language as string)}. {filteredOut} more are hidden; set the
            language to Auto-detect to see them.
          </p>
        )}
        {hidden > 0 && (
          <p className="border-t border-border px-3 py-2 text-xs text-muted-foreground">
            {hidden} more in the catalog are not installed. Pull them from Models.
          </p>
        )}
      </SelectContent>
    </Select>
  )
}
