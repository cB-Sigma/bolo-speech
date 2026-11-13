import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export interface Segment {
  key: string
  label: string
  value: number
  /** CSS colour. */
  color: string
  /** Formatted value shown in the legend and tooltip. */
  display: string
}

/** Thin part-to-whole bar with its figures listed beside it (the swatches are the legend). */
export function SegmentedBar({
  segments,
  className,
  ariaLabel,
}: {
  segments: Segment[]
  className?: string
  ariaLabel: string
}) {
  const total = segments.reduce((a, s) => a + Math.max(0, s.value), 0)
  const share = (v: number) => (total > 0 ? Math.max(0, v) / total : 0)
  const summary = segments.map((s) => `${s.label} ${s.display} (${Math.round(share(s.value) * 100)}%)`).join(', ')
  return (
    <div className={cn('min-w-0', className)}>
      <div
        role="img"
        aria-label={`${ariaLabel}: ${total > 0 ? summary : 'no data'}`}
        className="flex h-2 w-full gap-0.5 overflow-hidden rounded-full bg-muted"
      >
        {total > 0 &&
          segments
            .filter((s) => s.value > 0)
            .map((s) => (
              <Tooltip key={s.key}>
                <TooltipTrigger asChild>
                  <div
                    className="h-full min-w-[3px] transition-[flex-grow] duration-200 first:rounded-l-full last:rounded-r-full"
                    style={{ flexGrow: s.value, flexBasis: 0, background: s.color }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <span className="num font-medium">{s.display}</span> {s.label} · {Math.round(share(s.value) * 100)}%
                </TooltipContent>
              </Tooltip>
            ))}
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr_auto_auto] items-center gap-x-2 gap-y-1 text-xs">
        {segments.map((s) => (
          <div key={s.key} className="contents">
            <dt className="contents">
              <span className="size-2 rounded-[2px]" style={{ background: s.color }} aria-hidden />
              <span className="truncate text-muted-foreground">{s.label}</span>
            </dt>
            <dd className="contents">
              <span className="text-right num text-foreground">{s.display}</span>
              <span className="w-9 text-right num text-muted-foreground">
                {total > 0 ? `${Math.round(share(s.value) * 100)}%` : '–'}
              </span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
