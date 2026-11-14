import { Globe } from 'lucide-react'
import { useState } from 'react'
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
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { sharedEngine } from '@/lib/browser-stt/engine'
import {
  browserModel,
  BROWSER_MODELS,
  modelsByGroup,
  suggestedModelFor,
  type BrowserModel,
} from '@/lib/browser-stt/models'
import { peakFor, verdictFor, type DeviceInfo, type ModelNeeds } from '@/lib/device'
import { RiskDialog, useDevice, VerdictBadge } from '@/components/model-fit'
import { fmtBytes, languageName } from '@/lib/format'
import { groupLanguages, languageLabel } from '@/lib/languages'
import { cn } from '@/lib/utils'

/**
 * Language select with Indian languages listed first under their own names.
 * `auto` is offered as "Auto-detect" when there is more than one language.
 */
export function LanguageSelect({
  languages,
  value,
  onChange,
  disabled,
  className,
  autoDetect = true,
}: {
  languages: readonly string[]
  value: string
  onChange: (code: string) => void
  disabled?: boolean
  className?: string
  /** Offer "Auto-detect" (when false, the first language is the fallback). */
  autoDetect?: boolean
}) {
  const g = groupLanguages(languages)
  const count = g.indic.length + g.english.length + g.other.length
  const single = count === 1 ? (g.indic[0] ?? g.english[0] ?? g.other[0]) : null
  return (
    <Select value={single ? 'auto' : value} onValueChange={onChange} disabled={disabled || count === 0}>
      <SelectTrigger className={cn('h-8 w-40', className)} aria-label="Language" data-testid="language-select">
        <SelectValue />
      </SelectTrigger>
      <SelectContent
        position="popper"
        align="start"
        className="max-h-[min(26rem,var(--radix-select-content-available-height))]"
      >
        {single ? (
          <SelectItem value="auto">{languageLabel(single)}</SelectItem>
        ) : (
          <>
            {autoDetect && <SelectItem value="auto">Auto-detect</SelectItem>}
            {g.indic.length > 0 && (
              <SelectGroup>
                <SelectLabel className="eyebrow">Indian languages</SelectLabel>
                {g.indic.map((l) => (
                  <SelectItem key={l} value={l} textValue={languageName(l)}>
                    <span lang={l}>{languageLabel(l)}</span>
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
            {(g.english.length > 0 || g.other.length > 0) && (
              <SelectGroup>
                {g.indic.length > 0 && <SelectLabel className="eyebrow">Other languages</SelectLabel>}
                {[...g.english, ...g.other].map((l) => (
                  <SelectItem key={l} value={l}>
                    {languageName(l)}
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
          </>
        )}
      </SelectContent>
    </Select>
  )
}

function coverage(m: BrowserModel): string {
  return m.multilingual ? `${m.languages.length} languages` : languageName(m.languages[0] ?? 'en')
}

/** What a recogniser needs, for the device verdict. */
export function sttNeeds(m: BrowserModel, downloaded: boolean): ModelNeeds {
  return {
    bytes: m.approxBytes,
    peakBytes: peakFor(m.id, m.peakBytes),
    requiresWebGpu: m.requiresWebGpu,
    prefersWebGpu: m.prefersWebGpu,
    needsF16: m.requiresWebGpu,
    downloaded,
  }
}

function ModelOption({ m, downloaded, d }: { m: BrowserModel; downloaded: boolean; d: DeviceInfo }) {
  const verdict = verdictFor(sttNeeds(m, downloaded), d)
  return (
    <span className="flex min-w-0 flex-col">
      <span className="flex min-w-0 flex-wrap items-center gap-x-2 text-[13px] font-medium">
        {m.label}
        <span className="[[data-slot=select-trigger]_&]:hidden">
          <VerdictBadge verdict={verdict} />
        </span>
      </span>
      <span className="text-xs text-muted-foreground [[data-slot=select-trigger]_&]:hidden">
        {coverage(m)} · {fmtBytes(m.approxBytes)}
        {downloaded ? ' (downloaded)' : ' download'} · {m.license}
        {m.requiresWebGpu ? ' · needs WebGPU' : m.prefersWebGpu ? ' · best with WebGPU' : ''}
        {m.peakBytes > 900e6 ? ' · heavy for phones' : ''}
      </span>
      <span className="text-xs text-muted-foreground [[data-slot=select-trigger]_&]:hidden">
        {m.blurb}
        {m.licenseNote ? ` ${m.licenseNote}` : ''}
      </span>
    </span>
  )
}

/**
 * Picks which model the in-browser recogniser runs, grouped by language: name,
 * coverage, download size, licence, what it needs, and how it is expected to
 * fare on this device. A model not expected to work here asks first.
 */
export function BrowserModelPicker({
  value,
  onChange,
  disabled,
  className,
}: {
  value: string
  onChange: (id: string) => void
  disabled?: boolean
  className?: string
}) {
  const cachedIds = useStore(sharedEngine().store).cachedIds
  const d = useDevice()
  const [pending, setPending] = useState<BrowserModel | null>(null)
  const choose = (id: string) => {
    const m = browserModel(id)
    const verdict = verdictFor(sttNeeds(m, cachedIds.includes(id)), d)
    if (verdict.level === 'unlikely' || verdict.reasons.some((r) => r.includes('free in this browser'))) setPending(m)
    else onChange(id)
  }
  return (
    <>
      <Select value={value} onValueChange={choose} disabled={disabled}>
        <SelectTrigger
          className={cn('h-8 w-56 min-w-0', className)}
          aria-label="In-browser model"
          data-testid="browser-model-picker"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent
          position="popper"
          align="start"
          className="max-h-[min(30rem,var(--radix-select-content-available-height))] w-[min(26rem,calc(100vw-2rem))]"
        >
          {modelsByGroup().map((g) => (
            <SelectGroup key={g.group}>
              <SelectLabel className="eyebrow">{g.group}</SelectLabel>
              {g.models.map((m) => (
                <SelectItem
                  key={m.id}
                  value={m.id}
                  textValue={m.label}
                  className="py-1.5"
                  data-testid={`stt-model-${m.id}`}
                >
                  <ModelOption m={m} downloaded={cachedIds.includes(m.id)} d={d} />
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
      {pending && (
        <RiskDialog
          open
          label={pending.label}
          bytes={pending.approxBytes}
          verdict={verdictFor(sttNeeds(pending, cachedIds.includes(pending.id)), d)}
          onConfirm={() => {
            onChange(pending.id)
            setPending(null)
          }}
          onCancel={() => setPending(null)}
        />
      )}
    </>
  )
}

/**
 * Asked when a language is chosen that the current in-browser model cannot
 * do: which multilingual model to switch to, with the cost of each.
 */
export function SwitchModelDialog({
  language,
  current,
  onChoose,
  onCancel,
}: {
  /** The language that was asked for; null keeps the dialog closed. */
  language: string | null
  current: BrowserModel
  onChoose: (modelId: string) => void
  onCancel: () => void
}) {
  const cachedIds = useStore(sharedEngine().store).cachedIds
  const d = useDevice()
  const suggestedId = language ? suggestedModelFor(language)?.id : null
  const options = BROWSER_MODELS.filter((m) => language != null && m.languages.includes(language)).sort(
    (a, b) => Number(b.id === suggestedId) - Number(a.id === suggestedId),
  )
  const name = language ? languageName(language) : ''
  const suggested = language ? suggestedModelFor(language) : null
  return (
    <Dialog open={language != null} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent
        className="max-h-[calc(100svh-2rem)] overflow-y-auto sm:max-w-lg"
        data-testid="switch-model-dialog"
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Globe className="size-4 text-muted-foreground" aria-hidden />
            {name} needs a different model
          </DialogTitle>
          <DialogDescription>
            {current.label} only understands {languageName(current.languages[0] ?? 'en')}. These models run in your
            browser too. They are larger downloads and slower than Moonshine, and small speech models make many more
            mistakes in Indian languages than in English.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-2">
          {options.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => onChoose(m.id)}
              className="rounded-lg border border-border bg-background/60 px-3 py-2.5 text-left hover:bg-muted/60"
              data-testid={`switch-to-${m.id}`}
            >
              <span className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-sm font-medium">{m.label}</span>
                <VerdictBadge verdict={verdictFor(sttNeeds(m, cachedIds.includes(m.id)), d)} />
                {m.id === suggested?.id && (
                  <span className="rounded-md border border-border px-1.5 text-[11px] font-medium">Suggested</span>
                )}
                <span className="ml-auto num text-xs text-muted-foreground">
                  {cachedIds.includes(m.id) ? 'already downloaded' : `${fmtBytes(m.approxBytes)} download`} ·{' '}
                  {m.license}
                </span>
              </span>
              <span className="mt-0.5 block text-xs text-muted-foreground">{m.blurb}</span>
            </button>
          ))}
          {options.length === 0 && (
            <p className="text-sm text-muted-foreground">No in-browser model covers {name}. The bolo server does.</p>
          )}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onCancel}>
            Keep {current.label}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
