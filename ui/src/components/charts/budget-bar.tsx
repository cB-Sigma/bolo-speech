import { Check, TriangleAlert, X } from 'lucide-react'
import { budgetState, type BudgetState } from '@/lib/latency/stats'
import { fmtMs } from '@/lib/format'
import { cn } from '@/lib/utils'

const FILL: Record<BudgetState, string> = {
  good: 'bg-good',
  warning: 'bg-warning',
  bad: 'bg-critical',
  unknown: 'bg-chart-mute',
}
const TRACK: Record<BudgetState, string> = {
  good: 'bg-good/15',
  warning: 'bg-warning/20',
  bad: 'bg-critical/15',
  unknown: 'bg-muted',
}
const WORD: Record<BudgetState, string> = {
  good: 'Within budget',
  warning: 'Over budget',
  bad: 'Well over budget',
  unknown: 'No measurement yet',
}
const ICON = { good: Check, warning: TriangleAlert, bad: X, unknown: null } as const

/**
 * Meter of measured latency against a target. The axis runs to twice the
 * target so the tick sits mid-bar and overshoot stays visible.
 */
export function BudgetBar({
  measuredMs,
  targetMs,
  label = 'End to end',
  className,
}: {
  measuredMs: number | null
  targetMs: number
  label?: string
  className?: string
}) {
  const state = budgetState(measuredMs, targetMs)
  const scale = targetMs * 2
  const pct = measuredMs == null ? 0 : Math.min(100, (measuredMs / scale) * 100)
  const Icon = ICON[state]
  const delta = measuredMs == null ? null : measuredMs - targetMs
  return (
    <div className={cn('min-w-0', className)} data-testid="budget-bar" data-state={state}>
      <div className="mb-1.5 flex items-baseline justify-between gap-2 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="flex items-center gap-1.5">
          {Icon && (
            <Icon
              className={cn(
                'size-3.5',
                state === 'good' ? 'text-good-text' : state === 'warning' ? 'text-warning-text' : 'text-critical-text',
              )}
              aria-hidden
            />
          )}
          <span className="text-foreground">{WORD[state]}</span>
          {delta != null && (
            <span className="num text-muted-foreground">
              {delta > 0 ? '+' : '−'}
              {fmtMs(Math.abs(delta))}
            </span>
          )}
        </span>
      </div>
      <div
        role="meter"
        aria-label={`${label} latency against a ${targetMs} ms target`}
        aria-valuemin={0}
        aria-valuemax={scale}
        aria-valuenow={measuredMs == null ? 0 : Math.round(Math.min(measuredMs, scale))}
        aria-valuetext={measuredMs == null ? 'no measurement' : `${Math.round(measuredMs)} ms, target ${targetMs} ms`}
        className={cn('relative h-2 rounded-full', TRACK[state])}
      >
        <div
          className={cn('h-full rounded-full transition-[width] duration-200', FILL[state])}
          style={{ width: `${pct}%` }}
        />
        <div className="absolute -top-1 left-1/2 h-4 w-0.5 -translate-x-1/2 rounded-full bg-foreground" aria-hidden />
      </div>
      <div className="mt-1 flex justify-between num text-[11px] text-muted-foreground">
        <span>{measuredMs == null ? '–' : fmtMs(measuredMs)}</span>
        <span>target {fmtMs(targetMs)}</span>
      </div>
    </div>
  )
}
