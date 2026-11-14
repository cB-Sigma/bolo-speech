import { AlertTriangle, Download, Loader2, Play, RotateCcw, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useStore } from '@/hooks/use-store'
import { describeLoad, sharedEngine, type EngineState } from '@/lib/browser-stt/engine'
import { isOutOfMemory } from '@/lib/browser-runtime'
import { clearAllDownloads } from '@/lib/downloads'
import type { Store } from '@/lib/store'
import type { LoadStage } from '@/lib/browser-stt/models'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

const STAGES: { id: LoadStage; label: string }[] = [
  { id: 'runtime', label: 'Runtime' },
  { id: 'model', label: 'Model' },
  { id: 'compiling', label: 'Compile' },
  { id: 'warmup', label: 'Warm up' },
]

/** The load-related state both in-browser engines (recogniser and synthesiser) share. */
export type LoadState = Pick<
  EngineState,
  | 'status'
  | 'device'
  | 'threads'
  | 'files'
  | 'progress'
  | 'stage'
  | 'runtime'
  | 'stalled'
  | 'safeMode'
  | 'cached'
  | 'error'
> & { model: { label: string; approxBytes: number } }

/** An engine these components can show and drive: the recogniser by default, or the in-browser synthesiser. */
export interface LoadEngine {
  readonly store: Pick<Store<LoadState>, 'get' | 'subscribe'>
  load(): Promise<void>
  retrySafe(): Promise<void>
}

/** 0..100 across the whole load: downloads take most of the bar, the rest is split between compile and warm-up. */
export function overallPercent(s: LoadState): number {
  if (s.status === 'ready') return 100
  if (s.status !== 'loading') return 0
  const runtime = s.runtime && s.runtime.total > 0 ? Math.min(1, s.runtime.loaded / s.runtime.total) : 0
  switch (s.stage) {
    case 'runtime':
      return 2 + runtime * 28
    case 'model':
      return 30 + Math.min(1, s.progress) * 50
    case 'compiling':
      return 85
    case 'warmup':
      return 95
    default:
      return 1
  }
}

function runsOn(s: LoadState): string {
  return s.device === 'webgpu' ? 'WebGPU' : `WebAssembly · ${s.threads} thread${s.threads === 1 ? '' : 's'}`
}

/** One short line saying where the in-browser model is. */
export function modelStatusText(s: LoadState, noun = 'model'): string {
  switch (s.status) {
    case 'ready':
      return `Ready · ${runsOn(s)}`
    case 'loading':
      return describeLoad(s, fmtBytes, noun)
    case 'error':
      return 'Could not load'
    default:
      return s.cached ? 'Downloaded · loads when you start' : `Not downloaded · ${fmtBytes(s.model.approxBytes)}`
  }
}

/**
 * Always-visible status of the in-browser model. When nothing is loaded it
 * doubles as the button that fetches it.
 */
export function ModelStatusChip({
  className,
  engine = sharedEngine(),
  noun = 'model',
}: {
  className?: string
  engine?: LoadEngine
  noun?: string
}) {
  const s = useStore(engine.store)
  const idle = s.status === 'idle' || s.status === 'error'
  const body = (
    <>
      {s.status === 'loading' ? (
        <Loader2 className="size-3 shrink-0 animate-spin text-chart-1 motion-reduce:animate-none" aria-hidden />
      ) : (
        <span
          aria-hidden
          className={cn(
            'size-1.5 shrink-0 rounded-full',
            s.status === 'ready'
              ? 'bg-good'
              : s.status === 'error'
                ? 'bg-critical'
                : s.cached
                  ? 'bg-chart-1'
                  : 'border border-chart-axis',
          )}
        />
      )}
      <span className="min-w-0 truncate">
        <span className="font-medium text-foreground">{s.model.label}</span>
        <span className="text-muted-foreground"> · {modelStatusText(s, noun)}</span>
      </span>
      {idle && <Download className="size-3 shrink-0 text-muted-foreground" aria-hidden />}
    </>
  )
  const cls = cn(
    'inline-flex h-7 max-w-full min-w-0 items-center gap-1.5 rounded-full border border-border bg-background/60 px-2.5 text-xs',
    className,
  )
  return idle ? (
    <button
      type="button"
      className={cn(cls, 'cursor-pointer hover:bg-muted/60')}
      onClick={() => void engine.load().catch(() => undefined)}
      data-testid="model-chip"
      data-status={s.status}
      aria-label={`${s.model.label}: ${modelStatusText(s, noun)}. ${s.cached ? 'Load it now' : 'Download it now'}`}
    >
      {body}
    </button>
  ) : (
    <span className={cls} role="status" data-testid="model-chip" data-status={s.status}>
      {body}
    </span>
  )
}

/** Stage-by-stage progress of a load in flight. Renders nothing otherwise. */
export function LoadProgress({
  className,
  engine = sharedEngine(),
  noun = 'model',
}: {
  className?: string
  engine?: LoadEngine
  noun?: string
}) {
  const s = useStore(engine.store)
  const [elapsed, setElapsed] = useState(0)
  const loading = s.status === 'loading'
  useEffect(() => {
    if (!loading) return
    const started = performance.now()
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 500)
    return () => {
      clearInterval(timer)
      setElapsed(0)
    }
  }, [loading])
  if (!loading) return null
  const at = STAGES.findIndex((x) => x.id === s.stage)
  return (
    <div className={cn('min-w-0', className)} data-testid="load-progress" data-stage={s.stage ?? ''}>
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-medium" role="status" data-testid="load-stage">
          {describeLoad(s, fmtBytes, noun)}
        </span>
        <span className="shrink-0 num text-muted-foreground">{elapsed.toFixed(0)} s</span>
      </div>
      <Progress
        value={overallPercent(s)}
        aria-label={`${noun[0]?.toUpperCase()}${noun.slice(1)} load progress`}
        className="mt-1.5 h-1.5"
      />
      <ol className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground" aria-label="Load steps">
        {STAGES.map((st, i) => (
          <li
            key={st.id}
            aria-current={i === at ? 'step' : undefined}
            className={cn(
              'flex items-center gap-1',
              i < at && 'text-foreground',
              i === at && 'font-medium text-foreground',
            )}
          >
            <span
              aria-hidden
              className={cn(
                'size-1.5 rounded-full',
                i < at ? 'bg-good' : i === at ? 'bg-chart-1' : 'border border-chart-axis',
              )}
            />
            {st.id === 'model' ? `${noun[0]?.toUpperCase()}${noun.slice(1)}` : st.label}
            <span className="sr-only">{i < at ? ' (done)' : i === at ? ' (in progress)' : ' (waiting)'}</span>
          </li>
        ))}
      </ol>
      {s.stage === 'compiling' && (
        <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
          The download is finished. Compiling happens on this device and can take a while on a phone; it is quicker next
          time.
        </p>
      )}
    </div>
  )
}

/**
 * Shown when a load failed, or when nothing has moved for a while: the real
 * reason, and ways forward.
 */
export function LoadTrouble({
  onSample,
  className,
  engine = sharedEngine(),
  noun = 'model',
}: {
  onSample?: () => void
  className?: string
  engine?: LoadEngine
  /** What is being loaded, for the messages ("model", "voice"). */
  noun?: string
}) {
  const s = useStore(engine.store)
  const failed = s.status === 'error'
  const stalled = s.status === 'loading' && s.stalled != null
  if (!failed && !stalled) return null
  const what =
    s.stalled?.stage === 'runtime'
      ? 'downloading the runtime'
      : s.stalled?.stage === 'model'
        ? `downloading the ${noun}`
        : s.stalled?.stage === 'compiling'
          ? `compiling the ${noun}`
          : s.stalled?.stage === 'warmup'
            ? 'warming up'
            : 'starting'
  return (
    <div
      role="alert"
      className={cn('rounded-lg border border-warning/40 bg-warning/10 px-3 py-2.5 text-[13px] leading-5', className)}
      data-testid="load-trouble"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning-text" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="font-medium">
            {failed ? `The ${noun} could not be loaded` : `Still ${what} after ${s.stalled?.seconds} s`}
          </p>
          <p className="mt-0.5 break-words text-muted-foreground" data-testid="load-error">
            {failed
              ? s.error
              : s.stalled?.stage === 'compiling' || s.stalled?.stage === 'warmup'
                ? 'Nothing has been reported since the download finished. Slow devices can take this long; if it never ends, a single-threaded retry usually works.'
                : 'No data has arrived for a while. The network may be slow or blocked.'}
          </p>
          {failed && isOutOfMemory(s.error) && (
            <p className="mt-1.5 text-foreground" data-testid="load-oom">
              The browser would not give this page enough memory. Close other tabs (or the other in-browser model on
              this page), then retry. Phones and tablets limit the memory of each tab, and every open tab takes some.
            </p>
          )}
          {s.safeMode && failed && (
            <p className="mt-0.5 text-muted-foreground">
              {isOutOfMemory(s.error)
                ? 'It already retried on one thread with everything else released.'
                : 'That was the single-threaded attempt.'}
            </p>
          )}
          <div className="mt-2 flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => void engine.retrySafe().catch(() => undefined)}
              data-testid="retry-safe"
            >
              <RotateCcw /> Retry single-threaded
            </Button>
            {failed && isOutOfMemory(s.error) && (
              <Button
                size="sm"
                variant="outline"
                onClick={() =>
                  void clearAllDownloads().then(() =>
                    toast.success('Unloaded every model and removed the downloads', {
                      description: 'Pick a model and it downloads afresh.',
                    }),
                  )
                }
                data-testid="oom-clear"
              >
                <Trash2 /> Unload all and clear downloads
              </Button>
            )}
            {onSample && (
              <Button size="sm" variant="outline" onClick={onSample} data-testid="trouble-sample">
                <Play className="fill-current" /> Play a sample
              </Button>
            )}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Add <span className="num">?debug=1</span> to the address for a diagnostics panel you can copy.
          </p>
        </div>
      </div>
    </div>
  )
}
