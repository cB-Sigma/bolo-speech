import { cn } from '@/lib/utils'

/**
 * The bolo mark: level-meter bars hanging from a headline, the way letters
 * hang from the top line in Devanagari. The first bar doubles as the stem of a "b".
 */
export function LogoMark({ className, title }: { className?: string; title?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={cn('size-6', className)}
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <path d="M3.5 5.5h17" />
      <path d="M6.5 5.5v13" />
      <path d="M11 5.5v6.5" />
      <path d="M15.2 5.5v10" />
      <path d="M19.4 5.5v4.5" />
    </svg>
  )
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-2', className)}>
      <span className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
        <LogoMark className="size-[18px]" />
      </span>
      <span className="text-[17px] leading-none font-semibold tracking-[-0.03em]">bolo</span>
    </span>
  )
}
