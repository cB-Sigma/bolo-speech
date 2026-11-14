import { Layers, Scissors, Users } from 'lucide-react'
import { useMemo } from 'react'
import { Empty } from '@/components/panel'
import { fmtSeconds } from '@/lib/format'
import { separationSummary, speakersOf, talkTime, type DiarizedResult } from '@/lib/diarize'
import { cn } from '@/lib/utils'

/** Up to eight distinct speakers get palette colours; more share the neutral one. */
export function speakerColor(index: number): string {
  return index < 8 ? `var(--chart-${index + 1})` : 'var(--chart-mute)'
}

function clock(s: number): string {
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`
}

/**
 * A speaker-attributed transcript: a who-spoke-when strip (overlapping speech
 * and separation windows marked), talk time per speaker, then the segments.
 */
export function SpeakerResult({ result, className }: { result: DiarizedResult; className?: string }) {
  const speakers = useMemo(() => speakersOf(result.segments), [result.segments])
  const talk = useMemo(() => talkTime(result.segments), [result.segments])
  const duration = Math.max(
    0.1,
    result.duration ?? 0,
    ...result.segments.map((s) => s.end),
    ...(result.separation?.windows ?? []).map((w) => w.end),
  )
  const color = (sp: string | null | undefined) => (sp ? speakerColor(speakers.indexOf(sp)) : 'var(--chart-mute)')
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`
  const overlaps = result.segments.filter((s) => s.overlap).length
  const separated = result.segments.filter((s) => s.separated).length
  const sep = result.separation

  if (result.segments.length === 0) {
    return (
      <Empty icon={Users} title="No speech found" className={className}>
        {result.text || 'The recording has no transcribed segments.'}
      </Empty>
    )
  }

  return (
    <div className={cn('grid gap-4', className)} data-testid="speaker-result">
      {sep && (
        <p
          role="status"
          className={cn(
            'flex items-start gap-2 rounded-lg border px-3 py-2 text-[13px] leading-5',
            sep.status === 'applied' ? 'border-good/40 bg-good/10' : 'border-warning/50 bg-warning/10',
          )}
          data-testid="separation-note"
          data-status={sep.status}
        >
          <Scissors className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
          <span>
            <span className="font-medium">Separation {sep.status}.</span> {separationSummary(sep)}
            {sep.narrowband && ' The audio was telephone-band; separation is not tuned for it.'}
            {sep.time_ms != null && sep.time_ms > 0 && (
              <span className="text-muted-foreground">
                {' '}
                Took <span className="num">{Math.round(sep.time_ms)} ms</span>
                {sep.separated_audio_s ? (
                  <>
                    {' '}
                    for <span className="num">{fmtSeconds(sep.separated_audio_s)}</span> of audio
                  </>
                ) : null}
                .
              </span>
            )}
          </span>
        </p>
      )}

      {speakers.length > 0 && (
        <section aria-label="Who spoke when">
          <div className="relative grid gap-1.5" data-testid="speaker-timeline">
            {(sep?.windows ?? []).map((w, i) => (
              <div
                key={`w${i}`}
                aria-hidden
                className="pointer-events-none absolute inset-y-0 rounded-sm bg-chart-mute/20"
                style={{
                  left: `calc(2.5rem + (100% - 2.5rem) * ${w.start / duration})`,
                  width: `calc((100% - 2.5rem) * ${(w.end - w.start) / duration})`,
                }}
                title={`Separation window ${clock(w.start)}–${clock(w.end)}`}
              />
            ))}
            {speakers.map((sp, si) => (
              <div key={sp} className="grid grid-cols-[2.5rem_minmax(0,1fr)] items-center gap-0">
                <span className="num text-xs font-medium" style={{ color: speakerColor(si) }}>
                  {sp}
                </span>
                <div className="relative h-4 rounded-sm bg-muted/40">
                  {result.segments
                    .filter((s) => s.speaker === sp)
                    .map((s, i) => (
                      <div
                        key={i}
                        className={cn('absolute inset-y-0 rounded-[2px]', s.overlap && 'speaker-overlap')}
                        style={{
                          left: pct(s.start),
                          width: `max(2px, ${pct(s.end - s.start)})`,
                          backgroundColor: color(sp),
                          opacity: s.separated ? 0.65 : 1,
                        }}
                        title={`${sp} ${clock(s.start)}–${clock(s.end)}${s.overlap ? ', overlaps another speaker' : ''}${s.separated ? ', from a separated track' : ''}`}
                      />
                    ))}
                </div>
              </div>
            ))}
            <div className="grid grid-cols-[2.5rem_minmax(0,1fr)] text-[10px] text-muted-foreground">
              <span />
              <span className="flex justify-between num">
                <span>0:00</span>
                <span>{clock(duration)}</span>
              </span>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {speakers.map((sp, si) => (
              <span key={sp} className="inline-flex items-center gap-1.5">
                <span className="h-2 w-3 rounded-[2px]" style={{ background: speakerColor(si) }} aria-hidden />
                <span className="num text-foreground">{sp}</span> {fmtSeconds(talk.get(sp) ?? 0)}
              </span>
            ))}
            {overlaps > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-3 rounded-[2px] bg-chart-mute speaker-overlap" aria-hidden />
                {overlaps} overlapping
              </span>
            )}
            {(sep?.windows.length ?? 0) > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-3 rounded-[2px] bg-chart-mute/30" aria-hidden /> separation window
              </span>
            )}
          </div>
        </section>
      )}

      <ol className="grid gap-1.5" data-testid="speaker-segments">
        {result.segments.map((s, i) => (
          <li key={`${s.id ?? i}`} className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 text-[14px] leading-6">
            <span className="flex flex-col items-start pt-0.5">
              {s.speaker ? (
                <span
                  className="inline-flex h-5 min-w-6 items-center justify-center rounded-md px-1.5 num text-[11px] font-semibold text-white"
                  style={{ background: color(s.speaker) }}
                >
                  <span className="sr-only">Speaker </span>
                  {s.speaker}
                </span>
              ) : (
                <span className="text-xs text-muted-foreground">–</span>
              )}
            </span>
            <span className="min-w-0">
              <span className="mr-2 num text-[11px] text-muted-foreground">
                {clock(s.start)}–{clock(s.end)}
              </span>
              {s.overlap && (
                <span
                  className="mr-1.5 inline-flex items-center gap-1 rounded border border-warning/50 px-1 text-[11px] text-warning-text"
                  data-testid="overlap-badge"
                >
                  <Layers className="size-3" aria-hidden /> overlap
                </span>
              )}
              {s.separated && (
                <span
                  className="mr-1.5 inline-flex items-center gap-1 rounded border border-border px-1 text-[11px] text-muted-foreground"
                  data-testid="separated-badge"
                >
                  <Scissors className="size-3" aria-hidden /> separated
                </span>
              )}
              <span>{s.text || <span className="text-muted-foreground">(no words)</span>}</span>
            </span>
          </li>
        ))}
      </ol>
      {separated > 0 && (
        <p className="text-xs text-muted-foreground">
          {separated} segment{separated === 1 ? ' was' : 's were'} transcribed from a separated track; the same speaker
          may then have two segments in a row.
        </p>
      )}
    </div>
  )
}
