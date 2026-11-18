import { AudioWaveform, Cpu, Languages, Play, Repeat2, Sparkles } from 'lucide-react'
import { useEffect, useRef, type ReactNode } from 'react'
import { Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { useStore } from '@/hooks/use-store'
import { useSupport } from '@/hooks/use-support'
import { STT_RATE } from '@/lib/audio/capture'
import { audioModelsFor, NOT_IN_THE_PAGE, type AudioModel, type AudioTask } from '@/lib/browser-audio/models'
import { fmtBytes, fmtMs, fmtSeconds, languageName } from '@/lib/format'
import { prefs, setPref } from '@/lib/prefs'
import { DEFAULT_MAX_REPEATS } from '@/lib/repeat-guard'
import type { DenoisingTranscriber } from '@/lib/stt/denoising'
import type { LidDecision } from '@/lib/stt/routing'
import { Store } from '@/lib/store'

const REPEAT_CHOICES = [0, 2, 3, 4, 6, 8]

/** One option of the panel: an icon, a title with a line of explanation, and its control. */
export function OptionRow({
  icon,
  title,
  hint,
  control,
  children,
  testId,
}: {
  icon: ReactNode
  title: ReactNode
  hint: ReactNode
  control?: ReactNode
  children?: ReactNode
  testId?: string
}) {
  return (
    <div className="grid gap-2 py-3 first:pt-0 last:pb-0" data-testid={testId}>
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
        <div className="flex min-w-0 flex-1 basis-56 items-start gap-2.5">
          <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md border border-border bg-muted/50 text-muted-foreground [&_svg]:size-3.5">
            {icon}
          </span>
          <div className="min-w-0">
            <div className="text-[13px] leading-5 font-medium">{title}</div>
            <div className="text-xs leading-4 text-pretty text-muted-foreground">{hint}</div>
          </div>
        </div>
        {control && <div className="flex shrink-0 flex-wrap items-center gap-2">{control}</div>}
      </div>
      {children}
    </div>
  )
}

/** The repetition guard's limit, as the server's `max_repeats`. */
export function RepeatGuardOption({ disabled, removed }: { disabled?: boolean; removed?: number }) {
  const p = useStore(prefs)
  return (
    <OptionRow
      testId="repeat-guard"
      icon={<Repeat2 />}
      title="Repetition guard"
      hint={
        <>
          Whisper can loop on silence (“the the the …”). A phrase repeated back to back more than this many times is cut
          to one copy, as the server does (<span className="font-mono">max_repeats</span>, default {DEFAULT_MAX_REPEATS}
          ).
          {removed ? (
            <span className="text-foreground" data-testid="repeats-removed">
              {' '}
              This session: {removed} repeated word{removed === 1 ? '' : 's'} removed.
            </span>
          ) : null}
        </>
      }
      control={
        <Select
          value={String(p.browserMaxRepeats)}
          onValueChange={(v) => setPref('browserMaxRepeats', Number(v))}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-36" aria-label="Repetition guard">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {REPEAT_CHOICES.map((n) => (
              <SelectItem key={n} value={String(n)}>
                {n === 0 ? 'Off' : `${n}${n === DEFAULT_MAX_REPEATS ? ' (default)' : ''}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    />
  )
}

const OFF = 'off'

/** A model in a picker: its label, then id, size, licence (and a mark when it is not for commercial use). */
function ModelItem({ m }: { m: AudioModel }) {
  return (
    <SelectItem value={m.id} textValue={m.label}>
      <span className="flex min-w-0 flex-col">
        <span>
          {m.label}
          {m.commercial === false && <span className="text-warning-text"> · not for commercial use</span>}
        </span>
        <span className="text-xs text-muted-foreground">
          <span className="num">{m.id}</span> · {m.bytes ? fmtBytes(m.bytes) : 'nothing to download'} · {m.license}
        </span>
      </span>
    </SelectItem>
  )
}

/** A picker of the page's models for one task, with Off. */
export function AudioModelSelect({
  task,
  value,
  onChange,
  disabled,
  label,
  offLabel = 'Off',
  className = 'w-52',
}: {
  task: AudioTask
  value: string | null
  onChange: (id: string | null) => void
  disabled?: boolean
  label: string
  offLabel?: string | null
  className?: string
}) {
  return (
    <Select value={value ?? OFF} onValueChange={(v) => onChange(v === OFF ? null : v)} disabled={disabled}>
      <SelectTrigger size="sm" className={className} aria-label={label}>
        {/* The trigger shows the name only; the list shows id, size and licence too. */}
        <SelectValue>
          {value ? (audioModelsFor(task).find((m) => m.id === value)?.label ?? value) : offLabel}
        </SelectValue>
      </SelectTrigger>
      <SelectContent position="popper" align="end" className="w-[min(24rem,calc(100vw-2rem))]">
        {offLabel != null && <SelectItem value={OFF}>{offLabel}</SelectItem>}
        {audioModelsFor(task).map((m) => (
          <ModelItem key={m.id} m={m} />
        ))}
      </SelectContent>
    </Select>
  )
}

/** Server features of this task that the page leaves out, and why. */
function LeftOut({ task }: { task: AudioTask }) {
  const items = NOT_IN_THE_PAGE.filter((x) => x.task === task)
  if (!items.length) return null
  return (
    <details className="text-xs leading-4 text-muted-foreground">
      <summary className="cursor-pointer select-none">Not in the page</summary>
      <ul className="mt-1 grid list-disc gap-0.5 pl-4">
        {items.map((x) => (
          <li key={x.label}>
            <span className="text-foreground">{x.label}</span>: {x.why}
          </li>
        ))}
      </ul>
    </details>
  )
}

export function DeviceOption({ disabled }: { disabled?: boolean }) {
  const p = useStore(prefs)
  const support = useSupport()
  return (
    <OptionRow
      testId="audio-device"
      icon={<Cpu />}
      title="Where these models run"
      hint="Auto is WebAssembly: measured in Chromium, these models are too small to keep a GPU busy (each frame is a round trip). WebGPU is here to compare."
      control={
        <Select
          value={p.browserAudioDevice}
          onValueChange={(v) => setPref('browserAudioDevice', v as 'auto' | 'wasm' | 'webgpu')}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-48" aria-label="Where the audio models run">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">Auto (WebAssembly)</SelectItem>
            <SelectItem value="wasm">WebAssembly</SelectItem>
            <SelectItem value="webgpu" disabled={!support.webgpu}>
              WebGPU{support.webgpu ? '' : ' (not here)'}
            </SelectItem>
          </SelectContent>
        </Select>
      }
    />
  )
}

export function DenoiseOption({ disabled }: { disabled?: boolean }) {
  const p = useStore(prefs)
  return (
    <OptionRow
      testId="denoise-option"
      icon={<Sparkles />}
      title="Denoise first"
      hint={
        <>
          The audio goes through a denoiser before the recogniser hears it (the server’s{' '}
          <span className="font-mono">denoise</span>). It can hurt as well as help: on the server’s noisy test mixes it
          helped Moonshine and hurt Whisper. Strength mixes the denoised audio with the original.
        </>
      }
      control={
        <AudioModelSelect
          task="denoise"
          label="Denoiser"
          value={p.browserDenoise}
          onChange={(v) => setPref('browserDenoise', v)}
          disabled={disabled}
        />
      }
    >
      {p.browserDenoise && (
        <div className="flex items-center gap-3 pl-8.5">
          <span className="w-16 shrink-0 text-xs text-muted-foreground" id="denoise-strength-label">
            Strength
          </span>
          <Slider
            min={0.1}
            max={1}
            step={0.05}
            value={[p.browserDenoiseStrength]}
            onValueChange={([v]) => setPref('browserDenoiseStrength', v ?? 1)}
            disabled={disabled}
            aria-labelledby="denoise-strength-label"
            className="max-w-56 flex-1"
            data-testid="denoise-strength"
          />
          <span className="w-10 text-right num text-xs">{p.browserDenoiseStrength.toFixed(2)}</span>
        </div>
      )}
      <div className="pl-8.5">
        <LeftOut task="denoise" />
      </div>
    </OptionRow>
  )
}

export function LidOption({
  disabled,
  decision,
  onSuggest,
}: {
  disabled?: boolean
  decision?: LidDecision | null
  onSuggest?: (language: string) => void
}) {
  const p = useStore(prefs)
  return (
    <OptionRow
      testId="lid-option"
      icon={<Languages />}
      title="Identify the language first"
      hint={
        <>
          With the language on Auto, a file or sample is heard by a language identifier first and goes to a recogniser
          that speaks what it heard (the server’s <span className="font-mono">model=auto</span>). A recogniser not
          downloaded yet is suggested, never fetched by itself.
        </>
      }
      control={
        <AudioModelSelect
          task="lid"
          label="Language identifier"
          value={p.browserLid}
          onChange={(v) => setPref('browserLid', v)}
          disabled={disabled}
        />
      }
    >
      {decision && <LidResult decision={decision} onSuggest={onSuggest} />}
      <div className="pl-8.5">
        <LeftOut task="lid" />
      </div>
    </OptionRow>
  )
}

/** What the language identifier heard and what was done about it, in words and as the server's header. */
export function LidResult({ decision, onSuggest }: { decision: LidDecision; onSuggest?: (language: string) => void }) {
  const top = decision.ranked.slice(0, 3)
  const heard = decision.ranked[0]?.[0]
  return (
    <div
      className="grid gap-1 rounded-lg border border-border bg-muted/30 p-2.5 text-xs leading-4"
      data-testid="lid-result"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-medium">Heard</span>
        {top.map(([lang, prob]) => (
          <Badge key={lang} variant="outline" className="h-5 px-1.5 text-[11px]">
            {languageName(lang)} <span className="num">{Math.round(prob * 100)}%</span>
          </Badge>
        ))}
        <span className="text-muted-foreground">
          · {fmtSeconds(decision.seconds)} in <span className="num">{fmtMs(decision.ms)}</span>
        </span>
      </div>
      <p className="text-muted-foreground">{decision.summary}</p>
      {decision.recommended && heard && onSuggest && (
        <div>
          <Button size="xs" variant="outline" onClick={() => onSuggest(heard)} data-testid="lid-suggest">
            Choose a recogniser for {languageName(heard)}
          </Button>
        </div>
      )}
      <p className="font-mono text-[11px] break-all text-muted-foreground" data-testid="lid-route">
        {decision.header}
      </p>
    </div>
  )
}

const player = new Store<{ which: 'original' | 'denoised' | null }>({ which: null })

/** The take before and after the denoiser, to hear what it did. */
export function DenoiseCompare({ denoiser }: { denoiser: DenoisingTranscriber }) {
  const stats = useStore(denoiser.stats)
  const playing = useStore(player)
  const ctx = useRef<AudioContext | null>(null)
  useEffect(() => () => void ctx.current?.close(), [])
  if (!stats.seconds && !stats.error) return null
  const play = (which: 'original' | 'denoised') => {
    void ctx.current?.close()
    const takes = denoiser.takes()
    const data = takes[which]
    if (!data.length) return
    const c = new AudioContext()
    ctx.current = c
    const buf = c.createBuffer(1, data.length, STT_RATE)
    buf.getChannelData(0).set(data)
    const src = c.createBufferSource()
    src.buffer = buf
    src.connect(c.destination)
    src.onended = () => player.set({ which: null })
    player.set({ which })
    src.start()
  }
  return (
    <div className="grid gap-2 rounded-lg border border-border p-2.5" data-testid="denoise-compare">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <AudioWaveform className="size-3.5 text-muted-foreground" aria-hidden />
        <span className="font-medium">Before and after</span>
        <span className="text-muted-foreground" data-testid="denoise-stats">
          <span className="num">{stats.model}</span> at strength {stats.strength.toFixed(2)} ·{' '}
          {fmtSeconds(stats.seconds)} in <span className="num">{fmtMs(stats.ms)}</span>
          {stats.seconds > 0 && (
            <>
              {' '}
              (real-time factor <span className="num">{(stats.ms / 1000 / stats.seconds).toFixed(3)}</span>)
            </>
          )}{' '}
          on {stats.device}
        </span>
      </div>
      {stats.error && <p className="text-xs text-warning-text">Not denoised: {stats.error}</p>}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" onClick={() => play('original')} data-testid="play-original">
          <Play /> {playing.which === 'original' ? 'Playing original…' : 'Original'}
        </Button>
        <Button size="sm" variant="outline" onClick={() => play('denoised')} data-testid="play-denoised">
          <Play /> {playing.which === 'denoised' ? 'Playing denoised…' : 'Denoised'}
        </Button>
      </div>
    </div>
  )
}

/**
 * What the page does to the audio before the recogniser hears it and to the text after: the server's options for
 * transcriptions, run in this page.
 */
export function BrowserProcessingPanel({
  disabled,
  removed,
  denoiser,
  decision,
  onSuggest,
}: {
  disabled?: boolean
  removed?: number
  denoiser?: DenoisingTranscriber | null
  decision?: LidDecision | null
  onSuggest?: (language: string) => void
}) {
  return (
    <Panel
      title="Before and after recognition"
      description="The server's transcription options, run in this page"
      data-testid="browser-processing"
      bodyClassName="divide-y divide-border"
    >
      <DenoiseOption disabled={disabled} />
      {denoiser && (
        <div className="py-3">
          <DenoiseCompare denoiser={denoiser} />
        </div>
      )}
      <LidOption disabled={disabled} decision={decision} onSuggest={onSuggest} />
      <RepeatGuardOption disabled={disabled} removed={removed} />
      <DeviceOption disabled={disabled} />
    </Panel>
  )
}
