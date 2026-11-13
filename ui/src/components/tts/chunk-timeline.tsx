import { TriangleAlert } from 'lucide-react'
import { LegendKey } from '@/components/charts/chart-kit'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { fmtMs } from '@/lib/format'
import { niceStep } from '@/lib/latency/stats'
import type { TtsChunk } from '@/lib/tts/session'
import { cn } from '@/lib/utils'

/**
 * Each chunk on a shared clock (ms since the request): a tick where its audio
 * arrived and a bar where it plays. The distance between the two is how far
 * synthesis is ahead of playback; a bar that had to wait is an underrun.
 */
export function ChunkTimeline({ chunks, className }: { chunks: TtsChunk[]; className?: string }) {
  if (chunks.length === 0) return null
  const end = Math.max(...chunks.map((c) => Math.max(c.arrivedMs, (c.playsAtMs ?? c.arrivedMs) + c.dur * 1000)))
  const step = niceStep(end, 4)
  const scale = Math.ceil(end / step) * step
  const ticks = Array.from({ length: Math.round(scale / step) + 1 }, (_, i) => i * step)
  const pct = (ms: number) => `${Math.min(100, (ms / scale) * 100)}%`
  const cols = 'grid-cols-[1.25rem_minmax(0,1fr)] sm:grid-cols-[1.25rem_minmax(0,11rem)_minmax(0,1fr)_4.5rem]'
  return (
    <div className={cn('min-w-0', className)} data-testid="chunk-timeline">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-3 w-0.5 rounded-full bg-chart-2" aria-hidden /> audio arrived
        </span>
        <LegendKey color="var(--chart-1)" shape="rect" label="plays" />
        <span className="ml-auto max-sm:hidden">ms since request</span>
      </div>
      <div className={cn('grid items-end gap-x-3 border-b border-border pb-1 text-[11px] text-muted-foreground', cols)}>
        <span>#</span>
        <span className="max-sm:hidden">Text</span>
        <div className="relative h-4 max-sm:col-start-2">
          {ticks.map((t, i) => (
            <span
              key={t}
              className={cn(
                'absolute bottom-0 num',
                i === 0 ? '' : i === ticks.length - 1 ? '-translate-x-full' : '-translate-x-1/2',
              )}
              style={{ left: pct(t) }}
            >
              {t}
            </span>
          ))}
        </div>
        <span className="text-right max-sm:hidden">Headroom</span>
      </div>
      <ul>
        {chunks.map((c) => {
          const plays = c.playsAtMs ?? c.arrivedMs
          const headroom = plays - c.arrivedMs
          const late = c.lateMs > 0
          return (
            <li key={c.seq} className={cn('grid items-center gap-x-3 border-b border-border/60 py-1.5 text-xs', cols)}>
              <span className="num text-muted-foreground">{c.seq + 1}</span>
              <span className="truncate max-sm:col-start-2" title={c.text}>
                {c.text}
              </span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="relative h-5 max-sm:col-start-2"
                    tabIndex={0}
                    role="img"
                    aria-label={`Chunk ${c.seq + 1}: arrived at ${fmtMs(c.arrivedMs)}, plays from ${fmtMs(plays)} for ${fmtMs(c.dur * 1000)}${late ? `, playback waited ${fmtMs(c.lateMs)}` : ''}`}
                  >
                    {ticks.slice(1).map((t) => (
                      <div key={t} className="absolute inset-y-0 w-px bg-chart-grid" style={{ left: pct(t) }} />
                    ))}
                    <div
                      className="absolute top-1.5 h-2 rounded-[3px] bg-chart-1"
                      style={{ left: pct(plays), width: `max(3px, calc(${pct(c.dur * 1000)} - 2px))` }}
                    />
                    <div
                      className="absolute top-0.5 h-4 w-0.5 rounded-full bg-chart-2"
                      style={{ left: pct(c.arrivedMs) }}
                    />
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="grid gap-0.5">
                    <span>
                      arrived <span className="num font-medium">{fmtMs(c.arrivedMs)}</span>
                    </span>
                    <span>
                      plays <span className="num font-medium">{fmtMs(plays)}</span> for{' '}
                      <span className="num">{fmtMs(c.dur * 1000)}</span>
                    </span>
                    <span>
                      {late
                        ? `playback waited ${fmtMs(c.lateMs)}`
                        : `${fmtMs(Math.max(0, headroom))} ahead of playback`}
                    </span>
                  </div>
                </TooltipContent>
              </Tooltip>
              <span className="flex items-center justify-end gap-1 text-right num max-sm:hidden">
                {late ? (
                  <>
                    <TriangleAlert className="size-3 text-warning-text" aria-hidden />−{Math.round(c.lateMs)}
                  </>
                ) : (
                  `+${Math.round(Math.max(0, headroom))}`
                )}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
