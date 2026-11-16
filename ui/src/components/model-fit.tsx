import { Cpu, HardDrive, TriangleAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
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
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useStore } from '@/hooks/use-store'
import { oneRuntimeAtATime } from '@/lib/browser-runtime'
import {
  device,
  ensureDeviceProbe,
  platformLabel,
  refreshDeviceStorage,
  reprobeDevice,
  VERDICT_TEXT,
  verdictFor,
  type ModelNeeds,
  type Verdict,
} from '@/lib/device'
import { downloads, freeSpaceFor } from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'
import { readGpuPower, writeGpuPower, type GpuPower } from '@/lib/webgpu'

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
          ? `yes; 16-bit shaders ${gpu.f16 ? 'yes' : 'no'}; largest buffer ${MB(gpu.maxBufferSize)}`
          : 'no: models run on the CPU (WebAssembly)',
    ],
    ...(gpu?.available ? ([['GPU in use', gpuText(gpu.adapter, gpu.details)]] as [string, string][]) : []),
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
      {gpu?.available && <GpuChoice integrated={Boolean(gpu.details?.integrated)} windows={isWindows()} />}
      <p className="mt-2 text-xs text-muted-foreground">
        Pickers mark models that are heavy or unlikely to work here. These are estimates; every model stays available,
        and after a model has run once its real load time and memory on this device are remembered.
      </p>
    </Panel>
  )
}

function gpuText(
  label: string | null,
  details: { integrated: boolean; guessed?: boolean; fallback: boolean } | undefined,
): string {
  const what = label ?? 'not named by this browser'
  if (details?.fallback) return `${what} (the browser's software fallback, no GPU)`
  if (!details?.integrated) return what
  return details.guessed ? `${what} (most likely integrated graphics)` : `${what} (integrated graphics)`
}

function isWindows(): boolean {
  return /Windows/.test(globalThis.navigator?.userAgent ?? '')
}

/**
 * Which GPU the models ask for, and what to do when WebGPU got the integrated one. The request is a hint: Chrome
 * and Edge on Windows ignore it (crbug.com/369219127), and on a laptop with two GPUs Windows (or the NVIDIA
 * driver) decides per application, whatever the page asks.
 */
function GpuChoice({ integrated, windows }: { integrated: boolean; windows: boolean }) {
  const [power, setPower] = useState<GpuPower>(readGpuPower)
  const choose = (v: string) => {
    if (v !== 'high-performance' && v !== 'low-power') return
    setPower(v)
    writeGpuPower(v)
    void reprobeDevice()
  }
  return (
    <div className="mt-2 grid gap-2" data-testid="gpu-choice">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">GPU to ask for</span>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={power}
          onValueChange={choose}
          aria-label="GPU to ask for"
          className="flex-wrap justify-start"
        >
          <ToggleGroupItem value="high-performance" className="px-3 text-xs" data-testid="gpu-high-performance">
            High performance
          </ToggleGroupItem>
          <ToggleGroupItem value="low-power" className="px-3 text-xs" data-testid="gpu-low-power">
            Power saving
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <p className="text-xs text-muted-foreground" data-testid="gpu-choice-note">
        Kept in this browser and passed to the browser as a hint; &ldquo;GPU in use&rdquo; above is what it actually
        gave.{' '}
        {windows && (
          <>Chrome and Edge on Windows ignore the hint (crbug.com/369219127): Windows decides which GPU they get. </>
        )}
        Models loaded from now on use it; reload the page to move a model that is already loaded.
      </p>
      {integrated && power === 'high-performance' && (
        <p
          className="flex items-start gap-1.5 rounded-md border border-warning/50 p-2 text-xs leading-5"
          data-testid="gpu-integrated-warning"
        >
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning-text" aria-hidden />
          {windows ? (
            <span>
              WebGPU got the integrated GPU. Windows picks the GPU for the browser: Settings &gt; System &gt; Display
              &gt; Graphics &gt; (browser) &gt; High performance, then restart the browser. That helps when this
              computer also has a dedicated GPU (NVIDIA, or AMD Radeon RX); with an NVIDIA card, the NVIDIA Control
              Panel (Manage 3D settings &gt; Program Settings) can do the same.
            </span>
          ) : (
            <span>
              WebGPU got the integrated GPU although the page asked for the high-performance one. If this computer also
              has a dedicated GPU, the system chooses for the browser: set this browser to use the dedicated GPU in the
              system&apos;s graphics settings, then restart it.
            </span>
          )}
        </p>
      )}
    </div>
  )
}
