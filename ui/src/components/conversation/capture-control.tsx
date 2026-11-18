import { Mic } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import type { CaptureProcessing } from '@/lib/audio/capture'
import { cn } from '@/lib/utils'

const OPTIONS: { key: keyof CaptureProcessing; label: string; testId: string }[] = [
  { key: 'echoCancellation', label: 'Echo cancellation', testId: 'capture-aec' },
  { key: 'noiseSuppression', label: 'Noise suppression', testId: 'capture-ns' },
  { key: 'autoGainControl', label: 'Automatic gain', testId: 'capture-agc' },
]

/**
 * The browser's own processing of the microphone (`getUserMedia` constraints) for a conversation. While the agent
 * speaks, the browser's echo canceller and noise suppressor turn the user's voice down (double talk), which makes
 * interrupting the agent harder to hear; switched off, something else must remove the agent's voice from the
 * microphone: headphones, or the server's echo cancellation.
 */
export function CaptureControl({
  value,
  onChange,
  serverEcho,
  disabled,
  className,
}: {
  value: CaptureProcessing
  onChange: (p: CaptureProcessing) => void
  /** The server's echo cancellation is on for this session (chosen here, or its setting). */
  serverEcho: boolean
  disabled?: boolean
  className?: string
}) {
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="capture-control">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
        <span className="inline-flex items-center gap-1.5 font-medium">
          <Mic className="size-3.5 text-muted-foreground" aria-hidden /> Browser microphone processing
        </span>
        {OPTIONS.map((o) => (
          <label key={o.key} className="inline-flex items-center gap-1.5 text-muted-foreground">
            <Switch
              checked={value[o.key]}
              onCheckedChange={(on) => onChange({ ...value, [o.key]: on })}
              disabled={disabled}
              aria-label={o.label}
              data-testid={o.testId}
            />
            {o.label}
          </label>
        ))}
      </div>
      <p className="text-xs leading-4 text-muted-foreground">
        While the agent speaks, the browser’s echo canceller and noise suppressor turn your voice down, so interrupting
        it can go unheard; the server listens more closely while it speaks to make up for it. If interrupting still
        fails, turn noise suppression off first.
      </p>
      {!value.echoCancellation && !serverEcho && (
        <p className="text-xs leading-4 text-warning-text" data-testid="capture-no-aec">
          Without the browser’s echo canceller the agent hears itself through your speakers and interrupts its own
          answers. Use headphones, or turn on the server’s echo cancellation above.
        </p>
      )}
    </div>
  )
}
