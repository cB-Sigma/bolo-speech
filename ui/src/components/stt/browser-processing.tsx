import { Repeat2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { Panel } from '@/components/panel'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { prefs, setPref } from '@/lib/prefs'
import { DEFAULT_MAX_REPEATS } from '@/lib/repeat-guard'

const REPEAT_CHOICES = [0, 2, 3, 4, 6, 8]

/** One option of the panel: an icon, a title with a line of explanation, and its control. */
export function OptionRow({
  icon,
  title,
  hint,
  control,
  children,
  testId,
}: {
  icon: ReactNode
  title: ReactNode
  hint: ReactNode
  control?: ReactNode
  children?: ReactNode
  testId?: string
}) {
  return (
    <div className="grid gap-2 py-3 first:pt-0 last:pb-0" data-testid={testId}>
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
        <div className="flex min-w-0 flex-1 basis-56 items-start gap-2.5">
          <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md border border-border bg-muted/50 text-muted-foreground [&_svg]:size-3.5">
            {icon}
          </span>
          <div className="min-w-0">
            <div className="text-[13px] leading-5 font-medium">{title}</div>
            <div className="text-xs leading-4 text-pretty text-muted-foreground">{hint}</div>
          </div>
        </div>
        {control && <div className="flex shrink-0 flex-wrap items-center gap-2">{control}</div>}
      </div>
      {children}
    </div>
  )
}

/** The repetition guard's limit, as the server's `max_repeats`. */
export function RepeatGuardOption({ disabled, removed }: { disabled?: boolean; removed?: number }) {
  const p = useStore(prefs)
  return (
    <OptionRow
      testId="repeat-guard"
      icon={<Repeat2 />}
      title="Repetition guard"
      hint={
        <>
          Whisper can loop on silence (“the the the …”). A phrase repeated back to back more than this many times is cut
          to one copy, as the server does (<span className="font-mono">max_repeats</span>, default {DEFAULT_MAX_REPEATS}
          ).
          {removed ? (
            <span className="text-foreground" data-testid="repeats-removed">
              {' '}
              This session: {removed} repeated word{removed === 1 ? '' : 's'} removed.
            </span>
          ) : null}
        </>
      }
      control={
        <Select
          value={String(p.browserMaxRepeats)}
          onValueChange={(v) => setPref('browserMaxRepeats', Number(v))}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-32" aria-label="Repetition guard">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {REPEAT_CHOICES.map((n) => (
              <SelectItem key={n} value={String(n)}>
                {n === 0 ? 'Off' : `${n} repeats${n === DEFAULT_MAX_REPEATS ? ' (default)' : ''}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    />
  )
}

/**
 * What the page does to the audio before the recogniser hears it and to the text after: the server's options for
 * transcriptions, run in this page.
 */
export function BrowserProcessingPanel({ disabled, removed }: { disabled?: boolean; removed?: number }) {
  return (
    <Panel
      title="Before and after recognition"
      description="The server's transcription options, run in this page"
      data-testid="browser-processing"
      bodyClassName="divide-y divide-border"
    >
      <RepeatGuardOption disabled={disabled} removed={removed} />
    </Panel>
  )
}
