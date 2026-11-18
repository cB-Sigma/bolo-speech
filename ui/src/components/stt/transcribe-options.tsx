import { ChevronRight, Info, SlidersHorizontal, Waves } from 'lucide-react'
import { useId, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { CatalogSelect, SERVER_DEFAULT } from '@/components/catalog-select'
import { useInfo, useModels } from '@/hooks/use-api'
import { autoPull, isCloud } from '@/lib/capabilities'
import { parseRoute, type DiarizedResult } from '@/lib/diarize'
import { languageName } from '@/lib/format'
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

/**
 * Denoising before recognition: the server's default (off unless the server or the model's pin says otherwise),
 * off, or on with a strength and a denoise model. Opt-in on purpose: it helps some models on noisy audio and makes
 * others (Whisper) worse.
 */
export function DenoiseControl({
  value,
  onChange,
  disabled,
  live = false,
  className,
}: {
  value: Denoise
  onChange: (d: Denoise) => void
  disabled?: boolean
  /** A live stream: cloud denoisers enhance whole recordings only, so they are left out. */
  live?: boolean
  className?: string
}) {
  const id = useId()
  const info = useInfo()
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
          <CatalogSelect
            models={live ? denoisers.filter((m) => !isCloud(m)) : denoisers}
            value={value.model}
            onChange={(model) => onChange({ ...value, model })}
            label="Denoise model"
            defaultLabel="Default denoiser"
            disabled={disabled}
            autoPull={autoPull(info.data)}
            extra={(m) => (m.sample_rate && m.sample_rate !== 16000 ? `${m.sample_rate / 1000} kHz` : null)}
            className="w-56"
            testId="denoise-model"
          />
        </div>
      )}
      {value.mode === 'on' && (
        <p className="text-xs leading-4 text-warning-text">
          It can make clean audio, and Whisper&apos;s transcripts, worse; a strength below 1 mixes in the original.
          {live && ' Cloud denoisers work on whole recordings only, so they are not offered for a live stream.'}
        </p>
      )}
    </div>
  )
}

/**
 * Punctuation and capital letters restored after recognition: the server's default (off unless the server or the
 * model's pin says otherwise), off, auto (only for recognisers that write none) or on, with a punctuation model.
 */
export function PunctuationControl({
  value,
  onChange,
  disabled,
  className,
}: {
  /** '' server default, 'off', 'auto', 'on' or a punctuation model id (on with that model) */
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const info = useInfo()
  const models = useModels('punctuation').models
  const mode = value === '' ? 'default' : value === 'off' || value === 'auto' || value === 'on' ? value : 'on'
  const model = mode === 'on' && value !== 'on' ? value : ''
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="punctuation-control">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs" id={`${id}-label`}>
          Punctuation
        </Label>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={mode}
          onValueChange={(v) => v && onChange(v === 'default' ? '' : v)}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
        >
          <ToggleGroupItem value="default" className="px-2.5 text-xs">
            Server default
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="px-2.5 text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="auto" className="px-2.5 text-xs" data-testid="punctuation-auto">
            If missing
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="px-2.5 text-xs" data-testid="punctuation-on">
            Always
          </ToggleGroupItem>
        </ToggleGroup>
        {mode === 'on' && (
          <CatalogSelect
            models={models}
            value={model}
            onChange={(m) => onChange(m || 'on')}
            label="Punctuation model"
            defaultLabel="Default punctuation model"
            disabled={disabled}
            autoPull={autoPull(info.data)}
            className="w-56"
            testId="punctuation-model"
          />
        )}
      </div>
      {(mode === 'auto' || mode === 'on') && (
        <p className="text-xs leading-4 text-muted-foreground">
          Adds punctuation and capital letters after recognition, for the languages the punctuation model knows
          (English; Chinese and English). &ldquo;If missing&rdquo; leaves recognisers that write their own alone.
        </p>
      )}
    </div>
  )
}

/**
 * Inverse text normalisation after recognition: the server's default (off unless the server or the model's pin says
 * otherwise), off, or on with an `itn` model (by default the one for the transcript's language).
 */
export function ItnControl({
  value,
  onChange,
  disabled,
  className,
}: {
  /** '' server default, 'off', 'on' or an itn model id (on with that model) */
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const info = useInfo()
  const models = useModels('itn').models
  const mode = value === '' ? 'default' : value === 'off' ? 'off' : 'on'
  const model = mode === 'on' && value !== 'on' ? value : ''
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="itn-control">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs" id={`${id}-label`}>
          Numbers as figures
        </Label>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={mode}
          onValueChange={(v) => v && onChange(v === 'default' ? '' : v)}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
        >
          <ToggleGroupItem value="default" className="px-2.5 text-xs">
            Server default
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="px-2.5 text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="px-2.5 text-xs" data-testid="itn-on">
            On
          </ToggleGroupItem>
        </ToggleGroup>
        {mode === 'on' && (
          <CatalogSelect
            models={models}
            value={model}
            onChange={(m) => onChange(m || 'on')}
            label="Inverse normalisation model"
            defaultLabel="For the transcript's language"
            disabled={disabled}
            autoPull={autoPull(info.data)}
            extra={(m) => m.languages.map(languageName).join(', ') || null}
            className="w-56"
            testId="itn-model"
          />
        )}
      </div>
      {mode === 'on' && (
        <p className="text-xs leading-4 text-muted-foreground">
          &ldquo;twenty five dollars&rdquo; becomes &ldquo;$25&rdquo;: numbers, money, dates and times, with
          WeTextProcessing grammars (English, Chinese, Hindi; Apache-2.0). A transcript in another language is left as
          it was.
        </p>
      )}
    </div>
  )
}

/**
 * Redaction of personal information after recognition: the server's default (off unless the server or the model's
 * pin says otherwise), off, or on with a `pii` model and, optionally, the entity types to mask.
 */
export function RedactControl({
  value,
  entities,
  onChange,
  onEntitiesChange,
  disabled,
  className,
}: {
  /** '' server default, 'off', 'on' or a pii model id (on with that model) */
  value: string
  entities: string
  onChange: (v: string) => void
  onEntitiesChange: (v: string) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const info = useInfo()
  const models = useModels('pii').models
  const mode = value === '' ? 'default' : value === 'off' ? 'off' : 'on'
  const model = mode === 'on' && value !== 'on' ? value : ''
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="redact-control">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs" id={`${id}-label`}>
          Redact personal information
        </Label>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={mode}
          onValueChange={(v) => v && onChange(v === 'default' ? '' : v)}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
        >
          <ToggleGroupItem value="default" className="px-2.5 text-xs">
            Server default
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="px-2.5 text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="px-2.5 text-xs" data-testid="redact-on">
            On
          </ToggleGroupItem>
        </ToggleGroup>
        {mode === 'on' && (
          <CatalogSelect
            models={models}
            value={model}
            onChange={(m) => onChange(m || 'on')}
            label="Personal-information model"
            defaultLabel="Default model"
            disabled={disabled}
            autoPull={autoPull(info.data)}
            className="w-56"
            testId="redact-model"
          />
        )}
      </div>
      {mode === 'on' && (
        <>
          <Input
            value={entities}
            onChange={(e) => onEntitiesChange(e.target.value)}
            placeholder="All types it finds (or e.g. PERSON, PHONE_NUMBER)"
            aria-label="Entity types to mask"
            disabled={disabled}
            className="h-8 max-w-md text-xs"
            data-testid="redact-entities"
          />
          <p className="text-xs leading-4 text-muted-foreground">
            Each one found becomes [TYPE] in the text; the run notes say which model masked what. Patterns find numbers
            and e-mail addresses; names need the spaCy or GLiNER model.
          </p>
        </>
      )}
    </div>
  )
}

/** Names and terms the recogniser should favour (`hotwords`). */
export function HotwordsField({
  value,
  onChange,
  disabled,
  className,
}: {
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  return (
    <div className={cn('grid gap-1.5', className)}>
      <Label htmlFor={id} className="text-xs">
        Hotwords
      </Label>
      <Input
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Zintrava Labs, Kshitij Wadhwani"
        disabled={disabled}
        className="h-8 text-xs"
        data-testid="hotwords"
      />
      <p className="text-xs leading-4 text-muted-foreground">
        Names and terms to favour, comma separated. Whisper models (faster-whisper, whisper.cpp) and sherpa-onnx
        transducers (Zipformer, Parakeet TDT) use them; a model that cannot is named under &ldquo;Ignored&rdquo;.
      </p>
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
    <div className="min-w-0 rounded-lg border border-border" data-testid="decoding-options">
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
  const route = parseRoute(result.routeHeader)
  if (route) {
    items.push({
      label: 'Routed',
      value: `${route.requested} → ${route.answered}${route.reason ? ` (${route.reason})` : ''}${
        route.language ? ` · heard ${languageName(route.language)}` : ''
      }${route.probability != null ? ` ${(route.probability * 100).toFixed(0)} %` : ''}${route.lid ? ` by ${route.lid}` : ''}${
        route.recommended
          ? route.pulled
            ? ` · ${route.recommended} was not installed and was downloaded for this`
            : ` · better, not installed: ${route.recommended}`
          : ''
      }`,
      warn: route.reason === 'unsure' || route.reason === 'lid-failed' || route.reason === 'default',
      testId: 'run-route',
    })
  }
  if (result.languageHeader) {
    items.push({ label: 'x-bolo-language', value: result.languageHeader, testId: 'run-language-header' })
  }
  if (result.redactHeader) {
    items.push({ label: 'Redacted', value: result.redactHeader, testId: 'run-redact' })
  }
  if (result.itnHeader) {
    items.push({
      label: 'Numbers as figures',
      value: result.itnHeader,
      warn: result.itnHeader.startsWith('skipped'),
      testId: 'run-itn',
    })
  }
  if (result.punctuationHeader) {
    items.push({
      label: 'Punctuation',
      value: result.punctuationHeader,
      warn: result.punctuationHeader.startsWith('skipped'),
      testId: 'run-punctuation',
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
