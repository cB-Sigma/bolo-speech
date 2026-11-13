import { ChartLine, Table2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/** Shared Recharts axis styling: recessive hairlines, muted tabular tick labels. */
export const axisProps = {
  tickLine: false,
  axisLine: { stroke: 'var(--chart-axis)' },
  tick: { fill: 'var(--muted-foreground)', fontSize: 11, fontFamily: 'var(--font-mono)' },
  tickMargin: 6,
} as const

export const gridProps = { stroke: 'var(--chart-grid)', strokeDasharray: '0', vertical: false } as const

/** A chart card with a "view as table" twin, so no value is reachable only by hovering. */
export function ChartPanel({
  title,
  description,
  legend,
  chart,
  table,
  className,
  actions,
}: {
  title: string
  description?: ReactNode
  legend?: ReactNode
  chart: ReactNode
  table: ReactNode
  className?: string
  actions?: ReactNode
}) {
  const [asTable, setAsTable] = useState(false)
  return (
    <Panel
      title={title}
      description={description}
      className={className}
      actions={
        <>
          {actions}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-pressed={asTable}
                aria-label={asTable ? `Show ${title} as a chart` : `Show ${title} as a table`}
                onClick={() => setAsTable((v) => !v)}
              >
                {asTable ? <ChartLine /> : <Table2 />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{asTable ? 'Show chart' : 'Show as table'}</TooltipContent>
          </Tooltip>
        </>
      }
    >
      {legend && !asTable && (
        <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">{legend}</div>
      )}
      {asTable ? <div className="max-h-72 overflow-auto">{table}</div> : chart}
    </Panel>
  )
}

export function LegendKey({
  color,
  label,
  shape = 'line',
  dashed,
}: {
  color: string
  label: ReactNode
  shape?: 'line' | 'rect' | 'dot'
  dashed?: boolean
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {shape === 'line' ? (
        <span
          className={cn('h-0.5 w-4 rounded-full', dashed && 'opacity-70')}
          style={{ background: color }}
          aria-hidden
        />
      ) : shape === 'dot' ? (
        <span className="size-2 rounded-full" style={{ background: color }} aria-hidden />
      ) : (
        <span className="size-2.5 rounded-[2px]" style={{ background: color }} aria-hidden />
      )}
      {label}
    </span>
  )
}

export interface TipRow {
  color?: string
  label: string
  value: string
}

/** Tooltip body: values lead, labels follow, series keyed by a short stroke. */
export function TipBox({ title, rows, footer }: { title?: ReactNode; rows: TipRow[]; footer?: ReactNode }) {
  return (
    <div className="pointer-events-none min-w-36 rounded-lg border border-border bg-popover px-2.5 py-2 text-xs text-popover-foreground shadow-lg">
      {title && <div className="mb-1.5 num text-[11px] text-muted-foreground">{title}</div>}
      <div className="grid gap-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2">
            {r.color && (
              <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: r.color }} aria-hidden />
            )}
            <span className="num font-medium text-foreground">{r.value}</span>
            <span className="ml-auto pl-3 text-muted-foreground">{r.label}</span>
          </div>
        ))}
      </div>
      {footer && <div className="mt-1.5 border-t border-border pt-1.5 text-[11px] text-muted-foreground">{footer}</div>}
    </div>
  )
}

export function DataTable({ head, rows, caption }: { head: string[]; rows: (string | number)[][]; caption: string }) {
  return (
    <table className="w-full text-xs">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr className="text-left text-muted-foreground">
          {head.map((h, i) => (
            <th key={h} scope="col" className={cn('sticky top-0 bg-card py-1.5 font-medium', i > 0 && 'text-right')}>
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-t border-border">
            {r.map((c, j) => (
              <td key={j} className={cn('py-1.5 num', j > 0 && 'text-right')}>
                {c}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
