import type { ReactNode } from 'react'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { Sparkline } from './sparkline'

export type TileStatus = 'good' | 'warning' | 'bad' | 'unknown'

const STATUS_DOT: Record<TileStatus, string> = {
  good: 'bg-good',
  warning: 'bg-warning',
  bad: 'bg-critical',
  unknown: 'bg-chart-mute',
}
const STATUS_WORD: Record<TileStatus, string> = {
  good: 'within target',
  warning: 'near target',
  bad: 'over target',
  unknown: '',
}

/** Headline number with unit, optional context line, status and trend. */
export function StatTile({
  label,
  value,
  unit,
  context,
  status,
  trend,
  trendColor,
  loading,
  className,
}: {
  label: string
  value: ReactNode
  unit?: string
  context?: ReactNode
  status?: TileStatus
  trend?: readonly (number | null)[]
  trendColor?: string
  loading?: boolean
  className?: string
}) {
  return (
    <div
      className={cn('flex min-w-0 flex-col rounded-xl border border-border bg-card px-4 pt-3 pb-2.5', className)}
      data-testid="stat-tile"
    >
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span className="truncate">{label}</span>
      </div>
      {loading ? (
        <Skeleton className="mt-2 h-7 w-20" />
      ) : (
        <div className="mt-1 flex items-baseline gap-1">
          <span className="text-[26px] leading-8 font-semibold tracking-tight text-foreground">{value}</span>
          {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
        </div>
      )}
      <div className="mt-0.5 flex min-h-4 items-center gap-1.5 text-[11px] leading-4 text-muted-foreground">
        {status && status !== 'unknown' && (
          <>
            <span className={cn('size-1.5 shrink-0 rounded-full', STATUS_DOT[status])} aria-hidden />
            <span className="sr-only">{STATUS_WORD[status]}: </span>
          </>
        )}
        <span className="truncate">{context}</span>
      </div>
      {trend && <Sparkline values={trend} color={trendColor} className="mt-2" />}
    </div>
  )
}
