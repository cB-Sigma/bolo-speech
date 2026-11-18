import type { LabelScore } from '@/lib/api/types'

/** Classes with their probabilities as horizontal bars, most likely first. */
export function TagBars({ items, empty }: { items: LabelScore[]; empty: string }) {
  if (!items.length) return <p className="text-sm text-muted-foreground">{empty}</p>
  return (
    <ul className="grid gap-1.5" data-testid="tag-bars">
      {items.map((t) => (
        <li key={t.label} className="grid grid-cols-[minmax(0,10rem)_1fr_3rem] items-center gap-2 text-sm">
          <span className="truncate" title={t.label}>
            {t.label}
          </span>
          <span className="h-2 overflow-hidden rounded-full bg-muted" aria-hidden>
            <span
              className="block h-full rounded-full bg-link"
              style={{ width: `${Math.max(1, Math.round(t.score * 100))}%` }}
            />
          </span>
          <span className="num text-right text-xs text-muted-foreground">{Math.round(t.score * 100)} %</span>
        </li>
      ))}
    </ul>
  )
}
