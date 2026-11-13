import { AlertTriangle, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export function Panel({
  title,
  description,
  actions,
  children,
  className,
  bodyClassName,
  ...rest
}: {
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
} & Omit<React.ComponentProps<'section'>, 'title'>) {
  return (
    <section className={cn('flex min-w-0 flex-col rounded-xl border border-border bg-card', className)} {...rest}>
      {(title || actions) && (
        <header className="flex min-h-11 flex-wrap items-center gap-x-3 gap-y-1 border-b border-border px-4 py-2">
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[13px] leading-5 font-semibold">{title}</h2>
            {description && <p className="truncate text-xs leading-4 text-muted-foreground">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cn('min-w-0 flex-1 p-4', bodyClassName)}>{children}</div>
    </section>
  )
}

export function Empty({
  icon: Icon,
  title,
  children,
  action,
  className,
}: {
  icon?: LucideIcon
  title: string
  children?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-2 px-6 py-10 text-center', className)}>
      {Icon && (
        <span className="mb-1 flex size-9 items-center justify-center rounded-lg border border-border bg-muted/50 text-muted-foreground">
          <Icon className="size-4" aria-hidden />
        </span>
      )}
      <p className="text-sm font-medium">{title}</p>
      {children && <p className="max-w-sm text-[13px] leading-5 text-pretty text-muted-foreground">{children}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

export function ErrorState({
  title = 'Could not load this',
  error,
  onRetry,
  className,
}: {
  title?: string
  error?: string
  onRetry?: () => void
  className?: string
}) {
  return (
    <div
      role="alert"
      className={cn('flex flex-col items-center justify-center gap-2 px-6 py-10 text-center', className)}
    >
      <span className="mb-1 flex size-9 items-center justify-center rounded-lg border border-critical/30 bg-critical/10 text-critical-text">
        <AlertTriangle className="size-4" aria-hidden />
      </span>
      <p className="text-sm font-medium">{title}</p>
      {error && <p className="max-w-sm text-[13px] leading-5 text-muted-foreground">{error}</p>}
      {onRetry && (
        <Button size="sm" variant="outline" className="mt-2" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}

/** Small labelled value used in dense readouts. */
export function Readout({
  label,
  value,
  unit,
  hint,
  className,
}: {
  label: string
  value: ReactNode
  unit?: string
  hint?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('min-w-0', className)} data-testid="readout">
      <div className="truncate text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="num text-lg leading-6 font-medium text-foreground">{value}</span>
        {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 truncate text-[11px] leading-4 text-muted-foreground">{hint}</div>}
    </div>
  )
}
