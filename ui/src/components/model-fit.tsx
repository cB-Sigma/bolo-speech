import { Cpu, HardDrive, TriangleAlert } from 'lucide-react'
import { useEffect } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useStore } from '@/hooks/use-store'
import { oneRuntimeAtATime } from '@/lib/browser-runtime'
import {
  device,
  ensureDeviceProbe,
  platformLabel,
  refreshDeviceStorage,
  VERDICT_TEXT,
  verdictFor,
  type ModelNeeds,
  type Verdict,
} from '@/lib/device'
import { downloads, freeSpaceFor } from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

/** The device probe, run once and kept up to date as downloads change. */
export function useDevice() {
  const d = useStore(device)
  const list = useStore(downloads).entries
  useEffect(() => {
    void ensureDeviceProbe()
  }, [])
  useEffect(() => {
    void refreshDeviceStorage()
  }, [list])
  return d
}

export function useVerdict(needs: ModelNeeds): Verdict {
  return verdictFor(needs, useDevice())
}

/** A short label of how a model is expected to fare here; the reasons are in its accessible name and title. */
export function VerdictBadge({ verdict, className }: { verdict: Verdict; className?: string }) {
  if (verdict.level === 'ok') return null
  const text = verdict.level === 'heavy' ? 'Heavy here' : 'Unlikely here'
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 text-[11px] leading-5 font-medium',
        verdict.level === 'heavy' ? 'border-warning/50 text-warning-text' : 'border-destructive/50 text-destructive',
        className,
      )}
      title={verdict.reasons.join('; ')}
      data-testid="verdict-badge"
      data-level={verdict.level}
    >
      <TriangleAlert className="size-3" aria-hidden />
      {text}
    </span>
  )
}

/** One line before a download: size, and why it may struggle here. */
export function VerdictLine({ verdict, bytes, className }: { verdict: Verdict; bytes: number; className?: string }) {
  if (verdict.level === 'ok') return null
  return (
    <p
      className={cn('text-xs leading-5 text-muted-foreground', className)}
      data-testid="verdict-line"
      data-level={verdict.level}
    >
      <span className="font-medium text-foreground">{VERDICT_TEXT[verdict.level]}:</span> {verdict.reasons.join('; ')}.
      Download {fmtBytes(bytes)}. You can still use it.
    </p>
  )
}

/**
 * Asked before a model that is not expected to work on this device (or that
 * does not fit the storage left): the risk and the size, and the choice.
 */
export function RiskDialog({
  open,
  label,
  bytes,
  verdict,
  onConfirm,
  onCancel,
}: {
  open: boolean
  label: string
  bytes: number
  verdict: Verdict
  onConfirm: () => void
  onCancel: () => void
}) {
  const storage = verdict.reasons.some((r) => r.includes('free in this browser'))
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="sm:max-w-md" data-testid="risk-dialog">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TriangleAlert className="size-4 text-warning-text" aria-hidden />
            {label}: {VERDICT_TEXT[verdict.level].toLowerCase()}
          </DialogTitle>
          <DialogDescription>
            It is a {fmtBytes(bytes)} download. {verdict.reasons.join('; ')}. If the tab closes, the page notices on the
            next visit and loads nothing by itself.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="gap-2 sm:justify-between">
          <Button variant="ghost" onClick={onCancel}>
            Keep the current one
          </Button>
          <div className="flex flex-wrap gap-2">
            {storage && (
              <Button
                variant="outline"
                onClick={() =>
                  void freeSpaceFor(bytes).then((removed) => {
                    toast.message(
                      removed.length
                        ? `Removed ${removed.length} least recently used model${removed.length === 1 ? '' : 's'}`
                        : 'Nothing could be removed',
                    )
                    onConfirm()
                  })
                }
                data-testid="risk-free-space"
              >
                Free space and use it
              </Button>
            )}
            <Button onClick={onConfirm} data-testid="risk-confirm">
              Use it anyway
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

const MB = (b: number | null | undefined) => (b == null ? 'not reported' : fmtBytes(b))

/** What was detected about this device and browser, as the verdicts see it. */
export function DevicePanel({ className }: { className?: string }) {
  const d = useDevice()
  const gpu = d.webgpu
  const rows: [string, string][] = [
    ['Device', `${platformLabel(d.platform)} · ${d.browser}`],
    [
      'Memory',
      d.memoryGb != null ? `about ${d.memoryGb} GB (as the browser reports it)` : 'not reported by this browser',
    ],
    ['Processor', `${d.cores} thread${d.cores === 1 ? '' : 's'}`],
    [
      'WebGPU',
      gpu == null
        ? 'checking…'
        : gpu.available
          ? `yes${gpu.adapter ? ` (${gpu.adapter})` : ''}; 16-bit shaders ${gpu.f16 ? 'yes' : 'no'}; largest buffer ${MB(gpu.maxBufferSize)}`
          : 'no: models run on the CPU (WebAssembly)',
    ],
    ['WebAssembly threads', d.threads ? 'yes' : 'no (one thread): slower'],
    [
      'Storage',
      d.storage.quota != null
        ? `${MB(d.storage.usage)} used of ${MB(d.storage.quota)}${d.storage.persisted ? ', kept by the browser' : ', may be cleared by the browser when space is short'}`
        : 'not reported',
    ],
    ['Models at once', oneRuntimeAtATime() ? 'one (a phone, or short on memory)' : 'one recogniser and one voice'],
  ]
  return (
    <Panel
      title="Your device"
      description="What this browser offers the models; nothing is sent anywhere."
      className={className}
      data-testid="device-panel"
      bodyClassName="text-[13px] leading-5"
    >
      <dl className="grid grid-cols-[7.5rem_minmax(0,1fr)] gap-x-3 gap-y-1.5">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="flex items-center gap-1.5 text-muted-foreground">
              {k === 'Storage' ? (
                <HardDrive className="size-3.5" aria-hidden />
              ) : k === 'Processor' ? (
                <Cpu className="size-3.5" aria-hidden />
              ) : null}
              {k}
            </dt>
            <dd className="min-w-0 break-words">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-muted-foreground">
        Pickers mark models that are heavy or unlikely to work here. These are estimates; every model stays available,
        and after a model has run once its real load time and memory on this device are remembered.
      </p>
    </Panel>
  )
}
