import { Panel } from '@/components/panel'
import { fmtPercent } from '@/lib/format'
import type { WerResult } from '@/lib/latency/wer'

export interface RoundTrip {
  status: 'running' | 'done' | 'error'
  model: string
  hypothesis: string
  result: WerResult | null
  error?: string
  /** Where the recogniser ran, for the panel's description. */
  where?: string
}

export function WerDiff({ r, lang }: { r: WerResult; lang?: string }) {
  return (
    <p className="text-[15px] leading-8" data-testid="wer-diff" lang={lang}>
      {r.alignment.map((s, i) => {
        if (s.op === 'equal') return <span key={i}>{s.hyp} </span>
        if (s.op === 'insert')
          return (
            <span
              key={i}
              className="rounded-sm bg-chart-1/15 px-0.5 underline decoration-chart-1 decoration-2 underline-offset-4"
              title="Inserted: not in the text"
            >
              {s.hyp}
              <span className="sr-only"> (inserted)</span>{' '}
            </span>
          )
        if (s.op === 'delete')
          return (
            <span
              key={i}
              className="rounded-sm bg-critical/10 px-0.5 text-muted-foreground line-through decoration-critical"
              title="Missing from the transcript"
            >
              {s.ref}
              <span className="sr-only"> (missing)</span>{' '}
            </span>
          )
        return (
          <span key={i} title={`Expected “${s.ref}”`}>
            <span className="rounded-sm bg-critical/10 px-0.5 text-muted-foreground line-through decoration-critical">
              {s.ref}
            </span>
            <span className="rounded-sm bg-chart-1/15 px-0.5 underline decoration-chart-1 decoration-2 underline-offset-4">
              {s.hyp}
            </span>
            <span className="sr-only"> (substituted)</span>{' '}
          </span>
        )
      })}
    </p>
  )
}

/** The synthesised audio fed back into a recogniser, scored against the text. */
export function RoundTripPanel({ trip, lang }: { trip: RoundTrip; lang?: string }) {
  return (
    <Panel
      title="Round trip"
      description={
        <span>
          Synthesised audio transcribed by <span className="num">{trip.model}</span>
          {trip.where ? ` ${trip.where}` : ''}
        </span>
      }
      data-testid="round-trip-panel"
    >
      {trip.status === 'running' ? (
        <p className="text-sm text-muted-foreground" role="status">
          Synthesising, then transcribing…
        </p>
      ) : trip.status === 'error' ? (
        <p role="alert" className="text-sm text-critical-text">
          {trip.error}
        </p>
      ) : trip.result ? (
        <div className="grid gap-4 sm:grid-cols-[9rem_minmax(0,1fr)]">
          <div>
            <div className="text-xs text-muted-foreground">Word error rate</div>
            <div className="mt-0.5 text-3xl leading-9 font-semibold tracking-tight" data-testid="wer">
              {fmtPercent(trip.result.wer, trip.result.wer > 0 && trip.result.wer < 0.1 ? 1 : 0)}
            </div>
            <div className="mt-1 num text-[11px] leading-4 text-muted-foreground">
              {trip.result.substitutions} sub · {trip.result.deletions} del · {trip.result.insertions} ins
              <br />
              of {trip.result.refWords} words
            </div>
          </div>
          <div className="min-w-0">
            <WerDiff r={trip.result} lang={lang} />
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <span className="rounded-sm bg-critical/10 px-1 line-through decoration-critical">word</span> expected,
                not heard
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="rounded-sm bg-chart-1/15 px-1 underline decoration-chart-1 decoration-2 underline-offset-2">
                  word
                </span>{' '}
                heard instead
              </span>
            </div>
          </div>
        </div>
      ) : null}
    </Panel>
  )
}
