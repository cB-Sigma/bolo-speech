import { useId } from 'react'
import { cn } from '@/lib/utils'

/** Minimal trend line for stat tiles. Decorative: the tile's value carries the number. */
export function Sparkline({
  values,
  className,
  color = 'var(--chart-1)',
  min,
  max,
}: {
  values: readonly (number | null)[]
  className?: string
  color?: string
  min?: number
  max?: number
}) {
  const id = useId()
  const pts = values
    .map((v, i) => ({ v, i }))
    .filter((p): p is { v: number; i: number } => p.v != null && Number.isFinite(p.v))
  const W = 100
  const H = 28
  if (pts.length < 2)
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className={cn('h-7 w-full', className)} aria-hidden preserveAspectRatio="none" />
    )
  const lo = min ?? Math.min(...pts.map((p) => p.v))
  const hi = max ?? Math.max(...pts.map((p) => p.v))
  const span = hi - lo || 1
  const n = values.length - 1
  const x = (i: number) => (i / n) * W
  const y = (v: number) => H - 2 - ((v - lo) / span) * (H - 4)
  const line = pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.i).toFixed(2)} ${y(p.v).toFixed(2)}`).join(' ')
  const first = pts[0] as { i: number }
  const last = pts[pts.length - 1] as { i: number; v: number }
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={cn('h-7 w-full overflow-visible', className)}
      aria-hidden
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.18" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line} L${x(last.i)} ${H} L${x(first.i)} ${H} Z`} fill={`url(#${id})`} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
