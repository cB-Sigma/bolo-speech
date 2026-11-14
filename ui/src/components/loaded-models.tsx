import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Cpu, Hourglass, Infinity as InfinityIcon, MemoryStick, MonitorSmartphone, PowerOff, Timer } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useDevice } from '@/components/model-fit'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { tabMemoryBudget } from '@/lib/device'
import { KEEP_ALIVE_CHOICES, keepAlive, setKeepAliveMinutes } from '@/lib/keepalive'
import { toast } from 'sonner'
import { Empty, Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { describeError, useLive } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import { fmtBytes } from '@/lib/format'
import { clock, countdown, loadedRows, machineTotals, remaining, split, type LoadedRow } from '@/lib/loaded'
import { cn } from '@/lib/utils'

/** A thin meter: `value` of `max`, with an optional second share (bolo's part of a GPU's memory). */
function Meter({
  value,
  max,
  part,
  label,
  className,
}: {
  value: number | null
  max: number | null
  part?: number | null
  label: string
  className?: string
}) {
  const pct = value != null && max ? Math.min(100, (100 * value) / max) : 0
  const partPct = part != null && max ? Math.min(pct, (100 * part) / max) : 0
  return (
    <div
      className={cn('relative h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max ?? 0}
      aria-valuenow={value ?? 0}
    >
      <div className="absolute inset-y-0 left-0 bg-chart-2/50" style={{ width: `${pct}%` }} />
      {partPct > 0 && <div className="absolute inset-y-0 left-0 bg-chart-2" style={{ width: `${partPct}%` }} />}
    </div>
  )
}

/** CPU/GPU split of one model, by bytes. */
function SplitBar({ row }: { row: LoadedRow }) {
  const { gpu, cpu } = split(row)
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
      {gpu > 0 && <div className="bg-chart-2" style={{ width: `${gpu}%` }} />}
      {cpu > 0 && <div className="bg-chart-4/70" style={{ width: `${cpu}%` }} />}
    </div>
  )
}

function useTick(ms: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [ms])
  return now
}

const keepAliveLabel = (m: number | null) => (m == null ? 'Never' : m < 60 ? `${m} min` : `${m / 60} h`)

/** Phones, tablets and Apple-silicon Macs: the GPU uses the same memory as everything else. */
function sharedMemoryDevice(platform: string, adapter: string | null): boolean {
  return platform !== 'desktop' || /apple/i.test(adapter ?? '')
}

/**
 * The demo's totals: what the browser says about this device (a page cannot read its own CPU or memory use),
 * the models' estimated memory against what a tab may use here, and the in-browser keep-alive.
 */
function BrowserTotals({ rows }: { rows: LoadedRow[] }) {
  const d = useDevice()
  const ka = useStore(keepAlive)
  const budget = tabMemoryBudget(d)
  const used = rows.reduce((a, r) => a + r.size, 0)
  const onGpu = rows.reduce((a, r) => a + r.vram, 0)
  const gpu = d.webgpu
  const shared = sharedMemoryDevice(d.platform, gpu?.adapter ?? null)
  return (
    <div className="grid gap-3 border-b border-border px-4 py-3 text-xs" data-testid="machine-totals">
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 xl:grid-cols-3">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-x-2 text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <MemoryStick className="size-3.5" aria-hidden /> Models in this tab
            </span>
            <span className="num text-foreground" data-testid="browser-memory">
              {fmtBytes(used)} of about {fmtBytes(budget.bytes)}
              <span className="text-muted-foreground"> · estimate</span>
            </span>
          </div>
          <Meter
            value={used}
            max={budget.bytes}
            label="Estimated memory of the loaded models against this tab's limit"
          />
          <p className="mt-1 text-muted-foreground">Limit: {budget.why}.</p>
        </div>
        <div className="min-w-0" data-testid="browser-device">
          <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
            <MonitorSmartphone className="size-3.5" aria-hidden /> This device
          </div>
          <p className="text-foreground">
            {d.memoryGb != null ? (
              <>
                <span className="num">{d.memoryGb} GB</span> memory reported
                {d.memoryGb >= 8 && <span className="text-muted-foreground"> (browsers report at most 8)</span>}
              </>
            ) : (
              'Memory not reported by this browser'
            )}
            <span className="text-muted-foreground">
              {' '}
              · <span className="num">{d.cores}</span> cores{d.threads ? '' : ', one thread (no isolation)'}
            </span>
          </p>
        </div>
        <div className="min-w-0" data-testid="browser-gpu">
          <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
            <Cpu className="size-3.5" aria-hidden /> WebGPU
          </div>
          {gpu == null ? (
            <p className="text-muted-foreground">Checking…</p>
          ) : gpu.available ? (
            <p className="text-foreground">
              {gpu.adapter ?? 'Available'}
              {gpu.maxBufferSize != null && (
                <span className="text-muted-foreground">
                  {' '}
                  · largest buffer <span className="num">{fmtBytes(gpu.maxBufferSize)}</span>
                </span>
              )}
              {onGpu > 0 && (
                <span className="text-muted-foreground">
                  {' '}
                  · models on it <span className="num">{fmtBytes(onGpu)}</span>
                </span>
              )}
            </p>
          ) : (
            <p className="text-muted-foreground">Not offered here: every model runs on the CPU (WebAssembly).</p>
          )}
        </div>
      </div>
      {gpu?.available && shared && (
        <p className="text-muted-foreground" data-testid="shared-memory-note">
          On this device the GPU shares memory with everything else, so placing a model on WebGPU does not free memory:
          a CPU/GPU split here helps with WebGPU&apos;s buffer limits and with speed, not with total memory.
        </p>
      )}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <p className="min-w-0 text-muted-foreground">
          Memory figures are estimates from each model&apos;s size, or measured once on this device: a page cannot read
          one model&apos;s share.
        </p>
        <label className="flex items-center gap-2">
          <Hourglass className="size-3.5 text-muted-foreground" aria-hidden />
          <span className="text-muted-foreground">Unload after idle</span>
          <Select
            value={ka.minutes == null ? 'never' : String(ka.minutes)}
            onValueChange={(v) => setKeepAliveMinutes(v === 'never' ? null : Number(v))}
          >
            <SelectTrigger
              className="h-7 w-24"
              aria-label="Unload models after this long idle"
              data-testid="keep-alive"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {KEEP_ALIVE_CHOICES.map((m) => (
                <SelectItem key={String(m)} value={m == null ? 'never' : String(m)}>
                  {keepAliveLabel(m)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
      </div>
    </div>
  )
}

/**
 * Models the server holds in memory, Ollama-`ps` style: size, where each one runs (GPU, CPU or split), how
 * much accelerator memory it takes, and when it will be unloaded for being idle. Plus the machine's totals.
 */
export function LoadedModelsPanel({ className }: { className?: string }) {
  const feed = useLive()
  const qc = useQueryClient()
  const now = useTick(1000)
  const sampledAt = feed.history.at(-1)?.t ?? now
  const rows = loadedRows(feed.stats)
  const totals = machineTotals(feed.stats)

  const settle = () => void qc.invalidateQueries({ queryKey: ['catalog'] })
  const unload = useMutation({
    mutationFn: (id: string) => api.unload(id),
    onSuccess: (_r, id) => toast.success(`${id} unloaded`),
    onError: (err, id) => toast.error(`Could not unload ${id}`, { description: describeError(err) }),
    onSettled: settle,
  })
  const keep = useMutation({
    mutationFn: (id: string) => api.load(id, -1),
    onSuccess: (_r, id) => toast.success(`${id} stays loaded`, { description: 'Until you unload it.' }),
    onError: (err, id) => toast.error(`Could not keep ${id} loaded`, { description: describeError(err) }),
    onSettled: settle,
  })

  return (
    <Panel
      title="Loaded models"
      description="What is in memory now, where it runs and when it is unloaded"
      className={className}
      data-testid="loaded-models"
      bodyClassName="p-0"
    >
      {isDemo ? (
        <BrowserTotals rows={rows} />
      ) : (
        totals && (
          <div
            className="grid gap-x-6 gap-y-3 border-b border-border px-4 py-3 text-xs sm:grid-cols-2 xl:grid-cols-4"
            data-testid="machine-totals"
          >
            <div className="min-w-0">
              <div className="mb-1 flex items-center justify-between gap-2 text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <Cpu className="size-3.5" aria-hidden /> CPU
                </span>
                <span className="num text-foreground">{totals.cpuPercent.toFixed(0)}%</span>
              </div>
              <Meter value={totals.cpuPercent} max={100} label="CPU use" />
            </div>
            <div className="min-w-0">
              <div className="mb-1 flex items-center justify-between gap-2 text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <MemoryStick className="size-3.5" aria-hidden /> RAM
                </span>
                <span className="num text-foreground">
                  {totals.ramTotal
                    ? `${fmtBytes(totals.ramUsed)} of ${fmtBytes(totals.ramTotal)}`
                    : fmtBytes(totals.rss)}
                  <span className="text-muted-foreground"> · bolo {fmtBytes(totals.rss)}</span>
                </span>
              </div>
              <Meter value={totals.ramUsed ?? totals.rss} max={totals.ramTotal} part={totals.rss} label="Memory use" />
            </div>
            {totals.gpus.map((g) => (
              <div key={g.label} className="min-w-0" data-testid="gpu-total">
                <div className="mb-1 flex items-center justify-between gap-2 text-muted-foreground">
                  <span className="truncate" title={g.label}>
                    {g.label}
                    {g.busy != null && <span className="num"> · {g.busy.toFixed(0)}% busy</span>}
                  </span>
                  <span className="shrink-0 num text-foreground">
                    {g.total ? `${fmtBytes(g.used)} of ${fmtBytes(g.total)}` : 'memory unknown'}
                    {g.unified && <span className="text-muted-foreground"> · shared with RAM</span>}
                  </span>
                </div>
                <Meter value={g.used} max={g.total} part={g.bolo} label={`${g.label} memory`} />
              </div>
            ))}
            {totals.gpus.length === 0 && (
              <div className="self-end text-muted-foreground">No GPU memory readings: models run on the CPU</div>
            )}
          </div>
        )
      )}
      {rows.length === 0 ? (
        <Empty icon={MemoryStick} title="No models loaded" className="py-6">
          Models load on first use and unload after they have been idle for the keep-alive time.
        </Empty>
      ) : (
        <ul className="divide-y divide-border">
          {rows.map((row) => {
            const left = remaining(row.expiresIn, now - sampledAt)
            return (
              <li
                key={row.id}
                className="grid gap-x-4 gap-y-2 px-4 py-3 text-xs md:grid-cols-[minmax(0,2fr)_minmax(0,2fr)_auto_auto] md:items-center"
                data-testid="loaded-model"
                data-model={row.id}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-[13px] text-foreground">{row.id}</span>
                    <Badge variant="outline" className="h-4 px-1.5 text-[10px] uppercase">
                      {row.task}
                    </Badge>
                    {row.active > 0 && (
                      <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">
                        {row.active} active
                      </Badge>
                    )}
                  </div>
                  {row.notice && (
                    <p className="mt-0.5 truncate text-muted-foreground" title={row.notice}>
                      {row.notice}
                    </p>
                  )}
                </div>
                <div className="min-w-0">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="num text-foreground" data-testid="processor">
                      {row.processor}
                    </span>
                    <span className="num text-muted-foreground" data-testid="model-memory">
                      {fmtBytes(row.size)}
                      {row.estimated && ' (estimate)'}
                      {row.onGpu &&
                        (isDemo ? <> · WebGPU {fmtBytes(row.vram)}</> : <> · VRAM {fmtBytes(row.vram)}</>)}{' '}
                      · {row.device}
                    </span>
                  </div>
                  <SplitBar row={row} />
                  {row.onGpu && row.gpu?.total_bytes ? (
                    <Meter
                      className="mt-1"
                      value={row.vram}
                      max={row.gpu.total_bytes}
                      label={`${row.id}: share of ${row.gpu.name ?? 'GPU'} memory`}
                    />
                  ) : null}
                </div>
                <div
                  className="flex items-center gap-1.5 num text-muted-foreground md:justify-end"
                  data-testid="countdown"
                  title={row.keepAlive != null && row.keepAlive >= 0 ? `Keep-alive ${row.keepAlive} s` : undefined}
                >
                  {left == null && !row.stopping ? (
                    <InfinityIcon className="size-3.5" aria-hidden />
                  ) : (
                    <Timer className="size-3.5" aria-hidden />
                  )}
                  {countdown(left, row.stopping)}
                  {isDemo && (
                    <span data-testid="idle">
                      · idle {clock(row.idle + (row.active > 0 ? 0 : (now - sampledAt) / 1000))}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 md:justify-end">
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={row.expiresIn == null || keep.isPending}
                    onClick={() => keep.mutate(row.id)}
                    aria-label={`Keep loaded: ${row.id}`}
                  >
                    <InfinityIcon /> Keep loaded
                  </Button>
                  <Button
                    size="xs"
                    variant="ghost"
                    disabled={row.stopping || unload.isPending}
                    onClick={() => unload.mutate(row.id)}
                    aria-label={`Unload ${row.id}`}
                  >
                    <PowerOff /> Unload
                  </Button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}
