import { Check, Copy, ExternalLink, ServerOff } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { INSTALL_COMMAND, REPO_URL } from '@/lib/api/config'
import { cn } from '@/lib/utils'

/** The install command with a copy button. */
export function InstallCommand({ className }: { className?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(`${INSTALL_COMMAND}\nbolo serve`)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }
  return (
    <div
      className={cn(
        'flex min-w-0 items-stretch overflow-hidden rounded-lg border border-border bg-background/60',
        className,
      )}
    >
      <pre
        className="min-w-0 flex-1 overflow-x-auto px-3 py-2 num text-[12.5px] leading-5"
        tabIndex={0}
        aria-label="Install command"
      >
        <code>
          {INSTALL_COMMAND}
          {'\n'}bolo serve
        </code>
      </pre>
      <Button
        variant="ghost"
        size="sm"
        className="h-auto shrink-0 rounded-none border-l border-border px-3"
        onClick={() => void copy()}
      >
        {copied ? <Check /> : <Copy />}
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  )
}

/** Persistent strip under the header in the live demo. */
export function DemoBanner() {
  return (
    <aside
      aria-label="About this demo"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border bg-chart-1/10 px-4 py-2 text-[13px] sm:px-5"
      data-testid="demo-banner"
    >
      {/* The dot stays with the first line of the text at any width. */}
      <span className="flex min-w-0 flex-1 basis-64 items-baseline gap-2">
        <span className="size-1.5 shrink-0 -translate-y-0.5 rounded-full bg-chart-1" aria-hidden />
        <span className="min-w-0">
          <span className="font-medium">Live demo.</span> Everything here runs in your browser.{' '}
          <span className="text-muted-foreground">Install bolo to serve models over an API.</span>
        </span>
      </span>
      <a
        href={`${REPO_URL}#readme`}
        target="_blank"
        rel="noreferrer"
        className="ml-auto inline-flex items-center gap-1 rounded-sm font-medium text-link underline underline-offset-2"
      >
        Get bolo <ExternalLink className="size-3" aria-hidden />
      </a>
    </aside>
  )
}

/** Explains, in place, a feature the demo cannot offer, and how to get it. */
export function NeedsServer({
  title,
  children,
  className,
}: {
  title: string
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn('rounded-xl border border-dashed border-border bg-card p-5', className)}
      data-testid="needs-server"
    >
      <div className="flex items-start gap-3">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/50 text-muted-foreground">
          <ServerOff className="size-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold">{title}</h2>
          <div className="mt-1 max-w-2xl text-[13px] leading-5 text-muted-foreground">{children}</div>
          <InstallCommand className="mt-3 max-w-2xl" />
        </div>
      </div>
    </div>
  )
}

/** One-line reason shown next to a disabled control. */
export const SERVER_ONLY_HINT = 'Needs a bolo server. This demo runs in your browser only.'
