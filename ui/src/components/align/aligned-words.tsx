import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { AlignedWord } from '@/lib/api/types'
import { cn } from '@/lib/utils'

/** The word being spoken at `time` (seconds): the last one that has started, while it lasts or until the next. */
export function activeWord(words: AlignedWord[], time: number | null): number {
  if (time == null) return -1
  let at = -1
  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    if (w && w.start <= time) at = i
    else break
  }
  const w = at >= 0 ? words[at] : undefined
  const next = words[at + 1]
  // a pause after a word: nothing is highlighted until the next one starts
  if (w && time > w.end + 0.3 && (!next || next.start > time)) return -1
  return at
}

/**
 * Words with their times: the one being spoken at `time` is highlighted, a click plays from a word (`onSeek`).
 * A low alignment `score` (below 0.5) is underlined: the word may not be in the audio, or not where it was put.
 */
export function AlignedWords({
  words,
  time,
  onSeek,
  lang,
  className,
  testId = 'aligned-words',
}: {
  words: AlignedWord[]
  time: number | null
  onSeek?: (seconds: number) => void
  lang?: string
  className?: string
  testId?: string
}) {
  const at = activeWord(words, time)
  return (
    <p className={cn('text-[15px] leading-8', className)} lang={lang} data-testid={testId}>
      {words.map((w, i) => {
        const doubtful = w.score != null && w.score < 0.5
        return (
          <span key={i}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => onSeek?.(w.start)}
                  aria-current={i === at ? 'true' : undefined}
                  data-testid="aligned-word"
                  className={cn(
                    'rounded-sm px-0.5 transition-colors duration-100 hover:bg-chart-1/15 focus-visible:bg-chart-1/15 focus-visible:outline-none',
                    i === at && 'bg-chart-1/25 text-foreground',
                    doubtful && 'underline decoration-warning decoration-dotted underline-offset-4',
                  )}
                >
                  {w.word}
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <span className="num">
                  {w.start.toFixed(2)}–{w.end.toFixed(2)} s
                  {w.score != null && <> · score {Math.round(w.score * 100)} %</>}
                </span>
              </TooltipContent>
            </Tooltip>{' '}
          </span>
        )
      })}
    </p>
  )
}

/** Word times as WebVTT, one cue per word. */
export function wordsToVtt(words: AlignedWord[]): string {
  const stamp = (s: number) => {
    const ms = Math.max(0, Math.round(s * 1000))
    const h = Math.floor(ms / 3_600_000)
    const m = Math.floor((ms % 3_600_000) / 60_000)
    const sec = Math.floor((ms % 60_000) / 1000)
    const pad = (n: number, k = 2) => String(n).padStart(k, '0')
    return `${pad(h)}:${pad(m)}:${pad(sec)}.${pad(ms % 1000, 3)}`
  }
  return ['WEBVTT', '', ...words.map((w) => `${stamp(w.start)} --> ${stamp(w.end)}\n${w.word}\n`)].join('\n')
}
