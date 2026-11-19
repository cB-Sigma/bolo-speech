import { AlertTriangle, Download, Loader2, Play, RotateCcw, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useStore } from '@/hooks/use-store'
import { sharedEngine, type EngineState } from '@/lib/browser-stt/engine'
import { isOutOfMemory } from '@/lib/browser-runtime'
import { clearAllDownloads } from '@/lib/downloads'
import type { Store } from '@/lib/store'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

/** The load-related state both in-browser engines (recogniser and synthesiser) share. */
export { overallPercent } from '@/lib/load-progress'

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

function runsOn(s: LoadState): string {
  return s.device === 'webgpu' ? 'WebGPU' : `WebAssembly · ${s.threads} thread${s.threads === 1 ? '' : 's'}`
}

/** One short line saying where the in-browser model is. */
export function modelStatusText(s: LoadState, noun = 'model'): string {
  switch (s.status) {
    case 'ready':
      return `Ready · ${runsOn(s)}`
    case 'loading':
      // the stage and the bytes are in the downloads tray, the one place every download shows
      return s.stage === 'compiling' || s.stage === 'warmup' ? `Getting the ${noun} ready` : 'Loading'
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
