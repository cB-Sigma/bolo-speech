import { Check, Copy, Mic, MicOff, Play, RotateCcw, Upload } from 'lucide-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { detectEnvironment, micHelp, type Environment, type MicFailure } from '@/lib/mic-help'
import type { MicPermission } from '@/lib/support'
import { cn } from '@/lib/utils'

/**
 * Explains, in place, why the microphone did not open and what to do about
 * it on this device, with the alternatives right next to it.
 */
export function MicHelpPanel({
  failure,
  onRetry,
  onSample,
  onUpload,
  onDismiss,
  env,
  className,
}: {
  failure: MicFailure
  onRetry: () => void
  onSample?: () => void
  onUpload?: () => void
  onDismiss?: () => void
  env?: Environment
  className?: string
}) {
  const environment = useMemo(() => env ?? detectEnvironment(), [env])
  const help = micHelp(failure, environment)
  const [copied, setCopied] = useState(false)
  const url = globalThis.location?.href ?? ''
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.message('Copy this link', { description: url })
    }
  }
  return (
    <section
      aria-labelledby="mic-help-title"
      className={cn('rounded-xl border border-warning/40 bg-warning/10 p-4 text-[13px] leading-5 sm:p-5', className)}
      data-testid="mic-help"
      data-kind={failure.kind}
    >
      <div className="flex items-start gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-warning/40 bg-background/60 text-warning-text">
          <MicOff className="size-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <div role="alert">
            <h2 id="mic-help-title" className="text-sm font-semibold">
              {help.title}
            </h2>
            <p className="mt-1 max-w-2xl text-muted-foreground">{help.explanation}</p>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="sm" onClick={onRetry} data-testid="mic-retry">
              <RotateCcw /> Try again
            </Button>
            {help.offerCopyLink && (
              <Button size="sm" variant="outline" onClick={() => void copy()} data-testid="mic-copy-link">
                {copied ? <Check /> : <Copy />} {copied ? 'Link copied' : 'Copy link'}
              </Button>
            )}
            {onSample && (
              <Button size="sm" variant="outline" onClick={onSample} data-testid="mic-help-sample">
                <Play className="fill-current" /> Play a sample
              </Button>
            )}
            {onUpload && (
              <Button size="sm" variant="outline" onClick={onUpload} data-testid="mic-help-upload">
                <Upload /> Upload audio
              </Button>
            )}
            {onDismiss && (
              <Button size="sm" variant="ghost" onClick={onDismiss}>
                Dismiss
              </Button>
            )}
          </div>
          {help.offerCopyLink && (
            <p className="mt-2 num text-xs break-all text-muted-foreground" data-testid="mic-help-url">
              {url}
            </p>
          )}

          {help.sections.map((section, i) => (
            <details key={section.heading} open={i === 0} className="mt-3 max-w-2xl">
              <summary className="cursor-pointer rounded-sm text-[13px] font-medium">{section.heading}</summary>
              <ol className="mt-1.5 list-decimal space-y-1 pl-5 text-muted-foreground">
                {section.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </details>
          ))}

          <p className="mt-3 text-xs text-muted-foreground">
            What the browser reported:{' '}
            <span className="num">
              {failure.name || failure.kind}
              {failure.detail && failure.detail !== failure.name ? `: ${failure.detail}` : ''}
            </span>
            {failure.elapsedMs > 0 && (
              <>
                {' '}
                after <span className="num">{Math.round(failure.elapsedMs)} ms</span>
              </>
            )}
            {failure.permission !== 'unknown' && <> · saved permission: {failure.permission}</>}
          </p>
        </div>
      </div>
    </section>
  )
}

/** One line on the demo's first screen saying what will happen when the microphone button is pressed. */
export function MicStatusLine({
  permission,
  available,
  onShowHelp,
  className,
}: {
  permission: MicPermission
  /** False when the page cannot capture audio at all (no API, or not a secure context). */
  available: boolean
  onShowHelp: () => void
  className?: string
}) {
  const env = useMemo(() => detectEnvironment(), [])
  let dot = 'border border-chart-axis'
  let text: React.ReactNode
  if (!available) {
    dot = 'bg-critical'
    text = 'Microphone: not available in this browser. Play a sample or upload a file instead.'
  } else if (env.inApp) {
    dot = 'bg-warning'
    text = (
      <>
        Microphone: this looks like a browser inside {env.inApp}, where the microphone often does not work.{' '}
        <button type="button" className="rounded-sm text-link underline underline-offset-2" onClick={onShowHelp}>
          How to open it in {env.platform === 'ios' ? 'Safari' : env.platform === 'android' ? 'Chrome' : 'a browser'}
        </button>
      </>
    )
  } else if (permission === 'granted') {
    dot = 'bg-good'
    text = 'Microphone: allowed for this site. Recording starts as soon as you press the button.'
  } else if (permission === 'denied') {
    dot = 'bg-critical'
    text = (
      <>
        Microphone: blocked for this site.{' '}
        <button type="button" className="rounded-sm text-link underline underline-offset-2" onClick={onShowHelp}>
          How to allow it
        </button>
      </>
    )
  } else {
    text = 'Microphone: your browser will ask for permission when you press the button.'
  }
  return (
    <p
      className={cn('flex items-start gap-2 text-xs leading-5 text-muted-foreground', className)}
      data-testid="mic-status"
      data-permission={permission}
    >
      <span className="mt-[3px] flex shrink-0 items-center gap-1.5">
        <Mic className="size-3.5" aria-hidden />
        <span className={cn('size-1.5 rounded-full', dot)} aria-hidden />
      </span>
      <span>{text}</span>
    </p>
  )
}
