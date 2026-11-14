import { CircleCheck, CircleDashed, Lock } from 'lucide-react'
import type { PipelineId, PipelineOption } from '@/lib/capabilities'
import { cn } from '@/lib/utils'

/**
 * The three Realtime pipelines as a radio group. Unavailable ones stay visible
 * (greyed, with what they need) so the choice is explained, not hidden.
 */
export function PipelinePicker({
  options,
  value,
  onChange,
  locked,
  lockedReason = 'The pipeline is fixed once audio has started. End the conversation to change it.',
  className,
}: {
  options: PipelineOption[]
  value: PipelineId
  onChange: (id: PipelineId) => void
  locked?: boolean
  lockedReason?: string
  className?: string
}) {
  return (
    <fieldset className={cn('min-w-0', className)} data-testid="pipeline-picker">
      <legend className="sr-only">Pipeline</legend>
      <div className="grid gap-2 md:grid-cols-3">
        {options.map((o) => {
          const disabled = locked || !o.available
          const checked = value === o.id
          return (
            <label
              key={o.id}
              className={cn(
                'relative flex min-w-0 cursor-pointer flex-col gap-1 rounded-lg border p-3 text-left transition-colors duration-150',
                'has-[:focus-visible]:ring-[3px] has-[:focus-visible]:ring-ring/50',
                checked ? 'border-ring bg-chart-1/5' : 'border-border hover:border-input',
                disabled && 'cursor-not-allowed',
                !o.available && 'opacity-70',
              )}
              data-testid={`pipeline-${o.id}`}
              data-available={o.available}
            >
              <input
                type="radio"
                name="pipeline"
                value={o.id}
                checked={checked}
                disabled={disabled}
                onChange={() => onChange(o.id)}
                className="sr-only"
                aria-describedby={`pipeline-${o.id}-desc`}
              />
              <span className="flex items-center gap-2">
                <span
                  aria-hidden
                  className={cn(
                    'flex size-3.5 shrink-0 items-center justify-center rounded-full border',
                    checked ? 'border-ring' : 'border-input',
                  )}
                >
                  {checked && <span className="size-1.5 rounded-full bg-ring" />}
                </span>
                <span className="text-[13px] font-semibold">{o.label}</span>
                <span
                  className={cn(
                    'ml-auto inline-flex items-center gap-1 text-[11px]',
                    o.available ? 'text-good-text' : 'text-muted-foreground',
                  )}
                >
                  {o.available ? (
                    <CircleCheck className="size-3" aria-hidden />
                  ) : (
                    <CircleDashed className="size-3" aria-hidden />
                  )}
                  {o.available ? 'Available' : 'Not set up'}
                </span>
              </span>
              <span id={`pipeline-${o.id}-desc`} className="grid gap-1 text-xs leading-5 text-muted-foreground">
                <span>{o.summary}</span>
                {o.detail && <span className="num break-all text-foreground/80">{o.detail}</span>}
                {o.reason && <span>{o.reason}</span>}
                {o.note && <span className="text-warning-text">{o.note}</span>}
              </span>
              {locked && checked && (
                <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                  <Lock className="size-3" aria-hidden /> Locked for this conversation
                </span>
              )}
            </label>
          )
        })}
      </div>
      {locked && <p className="mt-2 text-xs text-muted-foreground">{lockedReason}</p>}
    </fieldset>
  )
}
