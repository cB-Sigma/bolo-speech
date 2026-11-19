import { Check, Download } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { useStore } from '@/hooks/use-store'
import { activeLoads, type ActiveLoad } from '@/lib/downloads'
import { LOAD_STEPS } from '@/lib/load-progress'
import { cn } from '@/lib/utils'

/**
 * Every model this page is fetching, in one place on every screen: the bottom left of the content area (toasts
 * take the bottom right). Each loader reports through `markLoading`, with its stage and bytes; pages show only a
 * one-word state ("Loading") and, when something goes wrong, what to do about it.
 */
export function DownloadTray({ className }: { className?: string }) {
  const loads = useStore(activeLoads)
  if (loads.length === 0) return null
  const running = loads.filter((l) => !l.done).length
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="download-tray"
      className={cn(
        'fixed bottom-[calc(1rem+env(safe-area-inset-bottom))] left-4 z-40 w-[min(22rem,calc(100vw-2rem))] rounded-xl border border-border bg-popover/95 p-3 text-popover-foreground shadow-lg backdrop-blur-md lg:left-60',
        className,
      )}
    >
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
        <Download className="size-3.5 text-link" aria-hidden />
        {running > 0 ? `Downloading ${running === 1 ? 'a model' : `${running} models`} into this browser` : 'Ready'}
      </div>
      <ul className="grid gap-3">
        {loads.map((l) => (
          <LoadRow key={l.id} load={l} />
        ))}
      </ul>
      {running > 0 && (
        <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
          Kept in this browser: next time it starts at once.
        </p>
      )}
    </div>
  )
}

function LoadRow({ load }: { load: ActiveLoad }) {
  const pct = Math.round(load.progress * 100)
  const at = LOAD_STEPS.findIndex((x) => x.id === load.stage)
  return (
    <li className="grid gap-1" data-testid={`download-${load.id}`} data-stage={load.stage ?? ''}>
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-medium">{load.label}</span>
        <span className="shrink-0 num text-muted-foreground">
          {load.done ? (
            <span className="inline-flex items-center gap-1 text-good-text">
              <Check className="size-3" aria-hidden /> ready
            </span>
          ) : (
            `${pct} %`
          )}
        </span>
      </div>
      <Progress value={pct} className="h-1.5" aria-label={`${load.label}: ${pct} %`} />
      {!load.done && load.detail && (
        <span className="truncate text-[11px] leading-4 text-muted-foreground" data-testid="download-detail">
          {load.detail}
          {load.stage === 'compiling' && ' · on this device, quicker next time'}
        </span>
      )}
      {!load.done && at >= 0 && (
        <ol className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground" aria-label="Load steps">
          {LOAD_STEPS.map((st, i) => (
            <li
              key={st.id}
              aria-current={i === at ? 'step' : undefined}
              className={cn('flex items-center gap-1', i <= at && 'text-foreground', i === at && 'font-medium')}
            >
              <span
                aria-hidden
                className={cn(
                  'size-1.5 rounded-full',
                  i < at ? 'bg-good' : i === at ? 'bg-chart-1' : 'border border-chart-axis',
                )}
              />
              {st.label}
              <span className="sr-only">{i < at ? ' (done)' : i === at ? ' (in progress)' : ' (waiting)'}</span>
            </li>
          ))}
        </ol>
      )}
    </li>
  )
}
