import { ArrowDown, ArrowUp, BookA } from 'lucide-react'
import { useId } from 'react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

/**
 * An ordered multi-select of word lists: ticked lists apply in the order shown, the first one winning where two
 * lists have the same word.
 */
export function WordListSelect({
  names,
  value,
  onChange,
  disabled,
  label = 'Word lists',
  className,
}: {
  names: string[]
  value: string[]
  onChange: (names: string[]) => void
  disabled?: boolean
  label?: string
  className?: string
}) {
  const id = useId()
  const others = names.filter((n) => !value.includes(n))
  const move = (i: number, d: -1 | 1) => {
    const next = value.slice()
    const j = i + d
    if (j < 0 || j >= next.length) return
    ;[next[i], next[j]] = [next[j] as string, next[i] as string]
    onChange(next)
  }
  const summary = value.length === 0 ? 'none' : value.length === 1 ? value[0] : `${value[0]} +${value.length - 1}`
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn('h-8 gap-1.5 font-normal', className)}
          disabled={disabled}
          aria-label={`${label}: ${value.length ? value.join(', ') : 'none'}`}
          data-testid="wordlist-select"
        >
          <BookA className="size-3.5" />
          {label}: <span className="num">{summary}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72">
        {names.length === 0 ? (
          <p className="text-xs text-muted-foreground">No word lists on this server yet. Create one on Word lists.</p>
        ) : (
          <div className="grid gap-2">
            <p className="text-xs leading-4 text-muted-foreground">
              Ticked lists apply in this order; the first wins when two have the same word.
            </p>
            <ul className="grid gap-1">
              {value.map((n, i) => (
                <li key={n} className="flex items-center gap-2">
                  <Checkbox id={`${id}-${n}`} checked onCheckedChange={() => onChange(value.filter((x) => x !== n))} />
                  <Label htmlFor={`${id}-${n}`} className="flex-1 num text-xs font-normal">
                    {i + 1}. {n}
                  </Label>
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    onClick={() => move(i, -1)}
                    disabled={i === 0}
                    aria-label={`Move ${n} up`}
                  >
                    <ArrowUp />
                  </Button>
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    onClick={() => move(i, 1)}
                    disabled={i === value.length - 1}
                    aria-label={`Move ${n} down`}
                  >
                    <ArrowDown />
                  </Button>
                </li>
              ))}
              {others.map((n) => (
                <li key={n} className="flex items-center gap-2">
                  <Checkbox id={`${id}-${n}`} checked={false} onCheckedChange={() => onChange([...value, n])} />
                  <Label htmlFor={`${id}-${n}`} className="num text-xs font-normal">
                    {n}
                  </Label>
                </li>
              ))}
            </ul>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
