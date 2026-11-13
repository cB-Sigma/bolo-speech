import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { fmtMs } from '@/lib/format'
import { layoutTimeline, utteranceLatency, type MarkerKind, type Utterance } from '@/lib/latency/timeline'
import { cn } from '@/lib/utils'

const MARKER_LABEL: Record<MarkerKind, string> = {
  speech_start: 'Speech starts',
  first_partial: 'First partial',
  speech_end: 'Speech ends',
  final: 'Final',
}

function describe(u: Utterance): string {
  const l = utteranceLatency(u)
  const parts: string[] = []
  if (l.speechMs != null) parts.push(`speech lasted ${fmtMs(l.speechMs)}`)
  if (l.firstPartialMs != null) parts.push(`first partial ${fmtMs(l.firstPartialMs)} after speech started`)
  if (l.finalMs != null) parts.push(`final ${fmtMs(l.finalMs)} after speech ended`)
  return parts.length ? parts.join('; ') : 'waiting for speech'
}

/**
 * One utterance on a time axis: the speech span, when the first partial
 * showed up, and how long after the end of speech the final arrived. Speech
 * is compressed so the gap after it, the part a user waits through, stays readable.
 */
export function LatencyTimeline({
  utterance,
  targetMs,
  className,
}: {
  utterance: Utterance | null
  targetMs?: number
  className?: string
}) {
  const layout = utterance ? layoutTimeline(utterance) : null
  const lat = utterance ? utteranceLatency(utterance) : null
  const pct = (p: number) => `${(p * 100).toFixed(2)}%`
  const endMarker = layout?.markers.find((m) => m.kind === 'speech_end')
  const finalMarker = layout?.markers.find((m) => m.kind === 'final')
  const tail =
    endMarker && finalMarker ? { pos0: endMarker.pos, pos1: finalMarker.pos, ms: finalMarker.at - endMarker.at } : null
  // Where the target would fall on the tail's scale.
  const targetPos = tail && targetMs && tail.ms > 0 ? tail.pos0 + ((tail.pos1 - tail.pos0) * targetMs) / tail.ms : null

  return (
    <div className={cn('min-w-0', className)} data-testid="latency-timeline">
      <div
        role="img"
        aria-label={utterance ? `Latency timeline: ${describe(utterance)}` : 'Latency timeline: no utterance yet'}
        className="relative h-14"
      >
        {/* axis */}
        <div className="absolute inset-x-0 top-7 h-px bg-chart-axis" />
        {layout?.speech && (
          <div
            className="absolute top-[22px] h-3 rounded-[3px] bg-chart-mute/45"
            style={{
              left: pct(layout.speech.pos0),
              width: pct(Math.max(0.004, layout.speech.pos1 - layout.speech.pos0)),
            }}
          />
        )}
        {tail && (
          <>
            <div
              className="absolute top-[26px] h-1 rounded-full bg-chart-1"
              style={{ left: pct(tail.pos0), width: pct(Math.max(0.004, tail.pos1 - tail.pos0)) }}
            />
            <div
              className="absolute top-0 -translate-x-1/2 num text-xs font-medium whitespace-nowrap text-foreground"
              style={{ left: pct(Math.min(0.94, Math.max(0.06, (tail.pos0 + tail.pos1) / 2))) }}
            >
              {fmtMs(tail.ms)}
            </div>
          </>
        )}
        {layout?.speech && (
          <div
            className="absolute top-0 -translate-x-1/2 text-[11px] whitespace-nowrap text-muted-foreground max-sm:hidden"
            style={{ left: pct((layout.speech.pos0 + layout.speech.pos1) / 2) }}
          >
            speech <span className="num">{lat?.speechMs != null ? `${(lat.speechMs / 1000).toFixed(1)} s` : ''}</span>
          </div>
        )}
        {targetPos != null && targetPos <= 1 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="absolute top-[18px] h-5 w-2 -translate-x-1/2" style={{ left: pct(targetPos) }}>
                <div className="mx-auto h-full w-px bg-foreground/50" />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              Target: <span className="num">{fmtMs(targetMs)}</span> after speech ends
            </TooltipContent>
          </Tooltip>
        )}
        {layout?.markers.map((m) => (
          <Tooltip key={m.kind}>
            <TooltipTrigger asChild>
              <div
                className="absolute top-[18px] flex h-5 w-5 -translate-x-1/2 items-center justify-center"
                style={{ left: pct(m.pos) }}
                tabIndex={0}
                aria-label={`${MARKER_LABEL[m.kind]} at ${fmtMs(m.at)}`}
              >
                {m.kind === 'first_partial' ? (
                  <span className="size-2.5 rounded-full bg-chart-2 ring-2 ring-card" />
                ) : m.kind === 'final' ? (
                  <span className="size-2.5 rounded-full bg-chart-1 ring-2 ring-card" />
                ) : (
                  <span className="h-4 w-0.5 rounded-full bg-foreground" />
                )}
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {MARKER_LABEL[m.kind]} · <span className="num">{fmtMs(m.at)}</span> from the start
            </TooltipContent>
          </Tooltip>
        ))}
        {!layout && (
          <div className="absolute inset-x-0 top-9 text-center text-xs text-muted-foreground">
            Speak, or play a sample, to see where the time goes.
          </div>
        )}
      </div>
      <dl className="grid grid-cols-3 gap-3 border-t border-border pt-3 text-xs">
        <div className="min-w-0">
          <dt className="flex items-center gap-1.5 text-muted-foreground">
            <span className="size-2 shrink-0 rounded-full bg-chart-2" aria-hidden />
            <span className="truncate">First partial</span>
          </dt>
          <dd className="mt-0.5">
            <span className="num text-sm font-medium text-foreground">{fmtMs(lat?.firstPartialMs)}</span>
            <span className="block truncate text-[11px] text-muted-foreground">after speech starts</span>
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="flex items-center gap-1.5 text-muted-foreground">
            <span className="size-2 shrink-0 rounded-full bg-chart-1" aria-hidden />
            <span className="truncate">Final</span>
          </dt>
          <dd className="mt-0.5">
            <span className="num text-sm font-medium text-foreground">{fmtMs(lat?.finalMs)}</span>
            <span className="block truncate text-[11px] text-muted-foreground">after speech ends</span>
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-3 shrink-0 rounded-[2px] bg-chart-mute/45" aria-hidden />
            <span className="truncate">Model time</span>
          </dt>
          <dd className="mt-0.5">
            <span className="num text-sm font-medium text-foreground">{fmtMs(utterance?.finalServerMs)}</span>
            <span className="block truncate text-[11px] text-muted-foreground">reported for the final</span>
          </dd>
        </div>
      </dl>
    </div>
  )
}

/**
 * All utterances of the session as bars from the end of speech (left edge) to
 * the final, on one shared scale so they can be compared, with the target marked.
 */
export function UtteranceList({
  utterances,
  targetMs,
  className,
}: {
  utterances: Utterance[]
  targetMs: number
  className?: string
}) {
  const done = utterances.filter((u) => u.finalAt != null)
  const rows = done.slice(-8).reverse()
  const lats = rows.map((u) => utteranceLatency(u))
  const max = Math.max(targetMs * 1.5, ...lats.map((l) => l.finalMs ?? 0)) * 1.08
  if (rows.length === 0) return null
  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-1 grid grid-cols-[1.25rem_minmax(0,1fr)_minmax(0,1.4fr)_4rem] items-center gap-x-3 text-[11px] text-muted-foreground">
        <span>#</span>
        <span>Utterance</span>
        <span>Speech end → final</span>
        <span className="text-right">ms</span>
      </div>
      <ul className="flex flex-col">
        {rows.map((u, i) => {
          const l = lats[i]
          const final = l?.finalMs ?? 0
          const partialAfterEnd =
            l?.endToFirstPartialMs != null && l.endToFirstPartialMs > 0 ? l.endToFirstPartialMs : null
          return (
            <li
              key={u.id}
              className="grid grid-cols-[1.25rem_minmax(0,1fr)_minmax(0,1.4fr)_4rem] items-center gap-x-3 border-t border-border py-1.5 text-xs"
            >
              <span className="num text-muted-foreground">{done.length - i}</span>
              <span className="truncate" title={u.text}>
                {u.text || '–'}
              </span>
              <div
                className="relative h-4"
                role="img"
                aria-label={`Final ${fmtMs(final)} after speech ended${final > targetMs ? ', over target' : ''}`}
              >
                <div className="absolute inset-y-0 left-0 w-px bg-chart-axis" />
                <div
                  className="absolute top-1.5 left-0 h-1 rounded-r-full bg-chart-1"
                  style={{ width: `${(final / max) * 100}%` }}
                />
                {partialAfterEnd != null && (
                  <span
                    className="absolute top-[3px] size-2.5 -translate-x-1/2 rounded-full bg-chart-2 ring-2 ring-card"
                    style={{ left: `${(partialAfterEnd / max) * 100}%` }}
                  />
                )}
                <span
                  className="absolute top-[3px] size-2.5 -translate-x-1/2 rounded-full bg-chart-1 ring-2 ring-card"
                  style={{ left: `${(final / max) * 100}%` }}
                />
                <div
                  className="absolute inset-y-0 w-px bg-foreground/40"
                  style={{ left: `${(targetMs / max) * 100}%` }}
                />
              </div>
              <span
                className={cn('text-right num', final > targetMs ? 'font-medium text-foreground' : 'text-foreground')}
              >
                {Math.round(final)}
              </span>
            </li>
          )
        })}
      </ul>
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        The vertical rule marks the <span className="num">{targetMs} ms</span> target.
      </p>
    </div>
  )
}
