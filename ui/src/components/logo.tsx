import { cn } from '@/lib/utils'
import { DEVANAGARI_BOLO } from './logo-devanagari'

/** A head in profile, drawn on a 32-unit grid; the neck runs off the bottom edge of the tile. */
const HEAD =
  'M6.4 33V25.2C4.2 22.9 3 19.8 3 16.4 3 9.6 8.3 4.4 14.6 4.4c3.9 0 6.6 1.9 7.6 5.1.4 1.3.5 2.4.5 3.4l2.3 3.6c.4.7.1 1.4-.7 1.6l-1.5.4v1.3l-4.3 1.4 4.1 1.5v1.9c0 1.9-1.2 3-3.1 3h-2.4V33z'

/**
 * The bolo mark: someone speaking. A head in profile with its mouth open, and
 * the voice leaving it as level-meter bars. Below 20 px the head is drawn
 * larger with two bars instead of three, so it still reads.
 */
export function LogoMark({
  className,
  title,
  small = true,
}: {
  className?: string
  title?: string
  /** Use the simplified drawing meant for small sizes. */
  small?: boolean
}) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={cn('size-7', className)}
      fill="currentColor"
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      {small ? (
        <>
          <path transform="translate(-1.6 -0.6) scale(0.92)" d={HEAD} />
          <rect x="24" y="15" width="2" height="8" rx="0.5" />
          <rect x="28" y="12" width="2" height="14" rx="0.5" />
        </>
      ) : (
        <>
          <path transform="translate(-1.2 0.6) scale(0.86)" d={HEAD} />
          <rect x="21.45" y="14.8" width="2.3" height="7" rx="1.15" />
          <rect x="24.75" y="11.8" width="2.3" height="13" rx="1.15" />
          <rect x="28.05" y="15.8" width="2.3" height="5" rx="1.15" />
        </>
      )}
    </svg>
  )
}

/** "बोलो" as outlines, sized by its height like text. */
function Devanagari({ className }: { className?: string }) {
  const { width, top, bottom, svg } = DEVANAGARI_BOLO
  return (
    <svg
      viewBox={`0 ${top} ${width} ${bottom - top}`}
      className={className}
      fill="currentColor"
      aria-hidden
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

/**
 * The lockup: tile, "bolo", and the same word in Devanagari ("बोलो", "speak").
 * `hindi="wide"` drops the Devanagari on narrow screens, where the header is tight.
 */
export function Logo({ className, hindi = 'always' }: { className?: string; hindi?: 'always' | 'wide' | 'never' }) {
  return (
    <span className={cn('inline-flex items-center gap-2', className)} role="img" aria-label="bolo">
      <span className="flex size-7 items-center justify-center overflow-hidden rounded-md bg-primary text-primary-foreground">
        <LogoMark className="size-7" />
      </span>
      <span className="text-[17px] leading-none font-semibold tracking-[-0.03em]" aria-hidden>
        bolo
      </span>
      {hindi !== 'never' && (
        <span
          className={cn('items-center gap-2 text-muted-foreground', hindi === 'wide' ? 'hidden sm:flex' : 'flex')}
          lang="hi"
          data-testid="logo-hindi"
        >
          <span className="size-[3px] rounded-full bg-current opacity-70" aria-hidden />
          <Devanagari className="h-[14.5px] w-auto translate-y-[0.5px]" />
        </span>
      )}
    </span>
  )
}
