import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { FinalSegment, TranscriptState } from '@/lib/stt/transcriber'
import { cn } from '@/lib/utils'

function Words({ seg }: { seg: FinalSegment }) {
  if (!seg.words || seg.words.length === 0) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            tabIndex={0}
            className="rounded-sm decoration-chart-mute decoration-dotted underline-offset-4 hover:underline focus-visible:underline"
          >
            {seg.text}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <span className="num">
            {seg.t0.toFixed(2)}–{seg.t1.toFixed(2)} s
          </span>
        </TooltipContent>
      </Tooltip>
    )
  }
  // Show the model's own punctuation and casing, aligning words by position.
  const display = seg.text.trim().split(/\s+/)
  const same = display.length === seg.words.length
  return (
    <>
      {seg.words.map((w, i) => (
        <span key={i}>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                tabIndex={0}
                className="rounded-sm transition-colors duration-150 hover:bg-chart-1/15 focus-visible:bg-chart-1/15"
              >
                {same ? display[i] : w.w}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <span className="num">
                {w.t0.toFixed(2)}–{w.t1.toFixed(2)} s
              </span>
            </TooltipContent>
          </Tooltip>
          {i < (seg.words?.length ?? 0) - 1 ? ' ' : ''}
        </span>
      ))}
    </>
  )
}

/**
 * Live transcript. Final text is solid; the partial's stable prefix is solid
 * too, while the tail that may still change is lighter and italic so the
 * difference survives without colour.
 */
export function Transcript({
  state,
  placeholder,
  className,
}: {
  state: TranscriptState | null
  placeholder: string
  className?: string
}) {
  const finals = state?.finals ?? []
  const partial = state?.partial ?? null
  const live = state?.status === 'streaming' || state?.status === 'finishing' || state?.status === 'reconnecting'
  const empty = finals.length === 0 && !partial?.text
  return (
    <div
      className={cn('min-h-28 text-[15px] leading-7 text-pretty', className)}
      aria-live="polite"
      aria-atomic="false"
      aria-label="Transcript"
      role="log"
      data-testid="transcript"
    >
      {empty && <p className="text-muted-foreground">{live ? 'Listening…' : placeholder}</p>}
      {finals.map((f, i) => (
        <span key={i} data-testid="final-text">
          <Words seg={f} />{' '}
        </span>
      ))}
      {partial?.text && (
        <span data-testid="partial-text">
          <span>{partial.text.slice(0, partial.stableChars)}</span>
          <span className="text-muted-foreground italic">{partial.text.slice(partial.stableChars)}</span>
        </span>
      )}
      {live && (
        <span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 live-dot rounded-full bg-chart-1" aria-hidden />
      )}
    </div>
  )
}
