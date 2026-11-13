import { Cpu, Download, HardDrive, Trash2, Zap } from 'lucide-react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useStore } from '@/hooks/use-store'
import type { SupportState } from '@/hooks/use-support'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { fmtBytes, fmtMs } from '@/lib/format'

function Fact({ label, ok, children }: { label: string; ok: boolean | null; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="flex items-center gap-1.5 text-right text-foreground">
        <span
          className={`size-1.5 rounded-full ${ok == null ? 'bg-chart-mute' : ok ? 'bg-good' : 'bg-chart-mute'}`}
          aria-hidden
        />
        {children}
      </span>
    </div>
  )
}

/** Status of the in-browser recogniser: download, cache, and what the browser offers it. */
export function BrowserEnginePanel({ support }: { support: SupportState }) {
  const engine = sharedEngine()
  const s = useStore(engine.store)
  const loaded = s.files.reduce((a, f) => a + f.loaded, 0)
  const total = s.files.reduce((a, f) => a + f.total, 0)

  const clear = async () => {
    engine.dispose()
    try {
      await caches.delete('transformers-cache')
      engine.store.patch({ cached: false })
      toast.success('Removed the downloaded model from this browser')
    } catch {
      toast.error('Could not clear the cache')
    }
  }

  return (
    <Panel
      title="In-browser model"
      description="Runs on this device with WebAssembly. Audio never leaves the page."
      data-testid="browser-engine"
      actions={
        s.status === 'idle' || s.status === 'error' ? (
          <Button size="sm" variant="outline" onClick={() => void engine.load().catch(() => undefined)}>
            <Download /> {s.cached ? 'Load' : 'Download'}
          </Button>
        ) : s.cached ? (
          <Button size="icon-sm" variant="ghost" aria-label="Remove downloaded model" onClick={() => void clear()}>
            <Trash2 />
          </Button>
        ) : null
      }
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate num text-[13px] font-medium">{s.model.label}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            English · {fmtBytes(s.model.approxBytes)} · {s.model.license} ·{' '}
            <a
              className="text-link underline underline-offset-2"
              href={s.model.source}
              target="_blank"
              rel="noreferrer"
            >
              source
            </a>
          </div>
        </div>
        <span
          className="shrink-0 rounded-md border border-border px-1.5 py-0.5 text-[11px] font-medium"
          data-testid="engine-status"
          data-status={s.status}
        >
          {s.status === 'ready'
            ? 'Ready'
            : s.status === 'loading'
              ? 'Loading'
              : s.status === 'error'
                ? 'Failed'
                : s.cached
                  ? 'Cached'
                  : 'Not downloaded'}
        </span>
      </div>

      {s.status === 'loading' && (
        <div className="mt-3">
          <Progress value={s.progress * 100} aria-label="Model download progress" className="h-1.5" />
          <div className="mt-1 flex justify-between num text-[11px] text-muted-foreground">
            <span>{total > 0 ? `${fmtBytes(loaded)} of ${fmtBytes(total)}` : 'Preparing…'}</span>
            <span>{s.progress >= 0.999 ? 'Compiling…' : `${Math.round(s.progress * 100)}%`}</span>
          </div>
        </div>
      )}
      {s.status === 'error' && (
        <p
          role="alert"
          className="mt-3 rounded-md border border-critical/30 bg-critical/10 px-2.5 py-2 text-xs text-foreground"
        >
          {s.error}
        </p>
      )}

      <div className="mt-3 divide-y divide-border border-t border-border">
        <Fact label="Runs on" ok={s.status === 'ready' ? true : null}>
          {s.device === 'webgpu' ? (
            <>
              <Zap className="size-3" aria-hidden /> WebGPU
            </>
          ) : s.device === 'wasm' ? (
            <>
              <Cpu className="size-3" aria-hidden /> WebAssembly · {s.threads} thread{s.threads === 1 ? '' : 's'}
            </>
          ) : support.webgpuAdapter ? (
            'WebGPU available'
          ) : (
            'WebAssembly'
          )}
        </Fact>
        <Fact label="SIMD" ok={support.wasmSimd}>
          {support.wasmSimd ? 'Available' : 'Unavailable'}
        </Fact>
        <Fact label="Threads" ok={support.wasmThreads}>
          {support.wasmThreads
            ? `Up to ${Math.min(4, support.hardwareConcurrency)}`
            : 'Off: page is not cross-origin isolated'}
        </Fact>
        {s.loadMs != null && (
          <Fact label="Load time" ok>
            <HardDrive className="size-3" aria-hidden /> <span className="num">{fmtMs(s.loadMs)}</span>
          </Fact>
        )}
      </div>
    </Panel>
  )
}
