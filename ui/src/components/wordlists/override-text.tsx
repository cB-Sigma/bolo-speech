import { splitOverrides } from '@/lib/wordlists'
import { cn } from '@/lib/utils'

/** Text with `[word](/ipa/)` overrides drawn as chips. */
export function OverrideText({ text, className }: { text: string; className?: string }) {
  return (
    <p className={cn('text-[14px] leading-7 break-words', className)} data-testid="preview-text">
      {splitOverrides(text).map((p, i) =>
        p.kind === 'text' ? (
          <span key={i}>{p.text}</span>
        ) : (
          <span
            key={i}
            className="mx-0.5 inline-flex items-baseline gap-1 rounded-md border border-chart-1/40 bg-chart-1/10 px-1.5 leading-6"
            title={`${p.word} is read as /${p.ipa}/`}
            data-testid="override-chip"
          >
            {p.word}
            <span className="num text-[12px] text-muted-foreground">/{p.ipa}/</span>
          </span>
        ),
      )}
    </p>
  )
}
