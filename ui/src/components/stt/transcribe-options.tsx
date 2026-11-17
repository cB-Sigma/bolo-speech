import { ChevronRight, Info, SlidersHorizontal, Waves } from 'lucide-react'
import { useId, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useModels } from '@/hooks/use-api'
import type { DiarizedResult } from '@/lib/diarize'
import {
  decodingCount,
  DECODING_FIELDS,
  emptyDecoding,
  VAD_FIELDS,
  type DecodingOptions,
  type Denoise,
  type TriState,
} from '@/lib/stt-options'
import { cn } from '@/lib/utils'

const SERVER_DEFAULT = '__default'

/**
 * Denoising before recognition: the server's default (off unless the server or the model's pin says otherwise),
 * off, or on with a strength and a denoise model. Opt-in on purpose: it helps some models on noisy audio and makes
 * others (Whisper) worse.
 */
export function DenoiseControl({
  value,
  onChange,
  disabled,
  className,
}: {
  value: Denoise
  onChange: (d: Denoise) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const denoisers = useModels('denoise').models
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="denoise-control">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs" id={`${id}-label`}>
          <Waves className="size-3.5 text-muted-foreground" aria-hidden /> Denoise first
        </Label>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={value.mode}
          onValueChange={(v) => v && onChange({ ...value, mode: v as Denoise['mode'] })}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
        >
          <ToggleGroupItem value="default" className="px-2.5 text-xs">
            Server default
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="px-2.5 text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="px-2.5 text-xs" data-testid="denoise-on">
            On
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      {value.mode === 'on' && (
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Strength</span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={value.strength}
              onChange={(e) => onChange({ ...value, strength: Number(e.target.value) })}
              disabled={disabled}
              className="w-28 accent-primary"
              aria-label="Denoise strength"
            />
            <span className="w-8 num">{value.strength.toFixed(2)}</span>
          </label>
          <Select
            value={value.model || SERVER_DEFAULT}
            onValueChange={(v) => onChange({ ...value, model: v === SERVER_DEFAULT ? '' : v })}
            disabled={disabled}
          >
            <SelectTrigger className="h-8 w-52" aria-label="Denoise model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              <SelectItem value={SERVER_DEFAULT}>Default denoiser</SelectItem>
              {denoisers.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  <span className="num">{m.id}</span>
                  <span className="text-xs text-muted-foreground [[data-slot=select-trigger]_&]:hidden">
                    {' '}
                    · {m.license}
                    {!m.installed ? ' · downloaded on first use' : ''}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {value.mode === 'on' && (
        <p className="text-xs leading-4 text-warning-text">
          It can make clean audio, and Whisper&apos;s transcripts, worse; a strength below 1 mixes in the original.
        </p>
      )}
    </div>
  )
}

function TriSelect({
  label,
  value,
  onChange,
  disabled,
  hint,
}: {
  label: string
  value: TriState
  onChange: (v: TriState) => void
  disabled?: boolean
  hint: string
}) {
  return (
    <label className="grid gap-1 text-xs">
      <span>{label}</span>
      <Select
        value={value || SERVER_DEFAULT}
        onValueChange={(v) => onChange(v === SERVER_DEFAULT ? '' : (v as TriState))}
      >
        <SelectTrigger className="h-8 w-full" aria-label={label} disabled={disabled}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
          <SelectItem value={SERVER_DEFAULT}>Default</SelectItem>
          <SelectItem value="true">On</SelectItem>
          <SelectItem value="false">Off</SelectItem>
        </SelectContent>
      </Select>
      <span className="text-[11px] leading-4 text-muted-foreground">{hint}</span>
    </label>
  )
}

/**
 * Whisper decoding options, the repetition guard and the silence removal, folded away by default. Empty fields
 * are the server's defaults; a model that cannot honour an option ignores it, and the result names it.
 */
export function DecodingSection({
  value,
  onChange,
  disabled,
}: {
  value: DecodingOptions
  onChange: (o: DecodingOptions) => void
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const count = decodingCount(value)
  return (
    <div className="rounded-lg border border-border" data-testid="decoding-options">
      <div className="flex items-center gap-2 pr-2">
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-left text-xs font-medium"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          data-testid="decoding-toggle"
        >
          <ChevronRight className={cn('size-3.5 shrink-0 transition-transform', open && 'rotate-90')} aria-hidden />
          <SlidersHorizontal className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
          Advanced decoding
          <span className="truncate font-normal text-muted-foreground">
            {count ? `${count} set` : 'server defaults'} · Whisper options, repetition guard, silence removal
          </span>
        </button>
        {count > 0 && (
          <Button size="xs" variant="ghost" onClick={() => onChange(emptyDecoding())} disabled={disabled}>
            Reset
          </Button>
        )}
      </div>
      {open && (
        <fieldset className="grid gap-3 border-t border-border p-3" disabled={disabled}>
          <p className="flex items-start gap-1.5 text-xs leading-4 text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            Empty means the default (the model&apos;s pinned setting, else the catalog&apos;s). faster-whisper honours
            all of these; whisper.cpp some; other models only the silence removal, chunk length and repetition guard.
            Against repetition loops: a penalty of 1.1 to 1.3, no repeated 3-grams, silence removal on.
          </p>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <TriSelect
              label="Condition on previous text"
              value={value.conditionOnPreviousText}
              onChange={(v) => onChange({ ...value, conditionOnPreviousText: v })}
              hint="Whisper; off by default: the main cause of loops on long audio"
            />
            {DECODING_FIELDS.map((f) => (
              <label key={f.key} className="grid gap-1 text-xs">
                <span>{f.label}</span>
                <Input
                  type="number"
                  inputMode="decimal"
                  step={f.step}
                  min={'min' in f ? f.min : undefined}
                  max={'max' in f ? f.max : undefined}
                  value={value.numbers[f.key] ?? ''}
                  onChange={(e) => onChange({ ...value, numbers: { ...value.numbers, [f.key]: e.target.value } })}
                  placeholder="default"
                  className="h-8 num"
                  aria-label={f.label}
                  data-testid={`opt-${f.key}`}
                />
                <span className="text-[11px] leading-4 text-muted-foreground">{f.hint}</span>
              </label>
            ))}
          </div>
          <div className="grid gap-3 border-t border-border pt-3 sm:grid-cols-2 lg:grid-cols-3">
            <TriSelect
              label="Silence removal (VAD filter)"
              value={value.vadFilter}
              onChange={(v) => onChange({ ...value, vadFilter: v })}
              hint="on for audio longer than the chunk length; the fields below switch it on"
            />
            {VAD_FIELDS.map((f) => (
              <label key={f.key} className="grid gap-1 text-xs">
                <span>{f.label}</span>
                <Input
                  type="number"
                  inputMode="decimal"
                  step={f.step}
                  min={f.min}
                  max={'max' in f ? f.max : undefined}
                  value={value.vad[f.key] ?? ''}
                  onChange={(e) => onChange({ ...value, vad: { ...value.vad, [f.key]: e.target.value } })}
                  placeholder="default"
                  className="h-8 num"
                  aria-label={`Silence removal: ${f.label}`}
                />
                <span className="text-[11px] leading-4 text-muted-foreground">{f.hint}</span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
    </div>
  )
}

/** What the server reported about a transcription beyond the text: language certainty, denoiser, ignored options. */
export function RunNotes({ result }: { result: DiarizedResult }) {
  const items: { label: string; value: string; warn?: boolean; testId: string }[] = []
  if (result.language) {
    items.push({
      label: 'Language',
      value:
        result.languageProbability != null
          ? `${result.language} · ${(result.languageProbability * 100).toFixed(0)} % sure`
          : result.language,
      warn: result.languageProbability != null && result.languageProbability < 0.5,
      testId: 'run-language',
    })
  }
  if (result.denoiseHeader) {
    items.push({
      label: 'Denoised',
      value: result.denoiseHeader,
      warn: result.denoiseHeader.startsWith('skipped'),
      testId: 'run-denoise',
    })
  }
  if (result.ignoredOptions) {
    items.push({ label: 'Ignored by this model', value: result.ignoredOptions, warn: true, testId: 'run-ignored' })
  }
  if (!items.length) return null
  return (
    <dl className="flex flex-wrap gap-x-5 gap-y-1 text-xs" data-testid="run-notes">
      {items.map((i) => (
        <div key={i.label} className="flex gap-1.5">
          <dt className="text-muted-foreground">{i.label}</dt>
          <dd className={cn('num', i.warn && 'text-warning-text')} data-testid={i.testId}>
            {i.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}
