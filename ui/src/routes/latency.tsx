import { Pause, Play } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from 'recharts'
import { axisProps, ChartPanel, DataTable, gridProps, LegendKey, TipBox } from '@/components/charts/chart-kit'
import { StatTile } from '@/components/charts/stat-tile'
import { RequestDrawer, Waterfall } from '@/components/latency/waterfall'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useLive } from '@/hooks/use-api'
import { isDemo } from '@/lib/api/config'
import { useStore } from '@/hooks/use-store'
import type { ModelStats, Percentiles, RequestTrace } from '@/lib/api/types'
import { fmtAudioTotal, fmtBytes, fmtClock, fmtInt, fmtMs, fmtRtf, fmtSpeed } from '@/lib/format'
import { bucketize, filterTraces, METRIC_LABEL, metricValue, type Metric } from '@/lib/latency/series'
import { budgetState, histogram, niceStep, percentile, summarize } from '@/lib/latency/stats'
import type { LiveState } from '@/lib/live'
import { prefs } from '@/lib/prefs'
import { cn } from '@/lib/utils'
import { Activity } from 'lucide-react'

const RANGES = [
  { value: '60', label: '1 min' },
  { value: '300', label: '5 min' },
  { value: '900', label: '15 min' },
]
const ALL = '__all'

const lineConfig = {
  p50: { label: 'p50', color: 'var(--chart-1)' },
  p95: { label: 'p95', color: 'var(--chart-2)' },
} satisfies ChartConfig
const histConfig = { count: { label: 'Requests', color: 'var(--chart-1)' } } satisfies ChartConfig

const ms = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '–' : String(Math.round(v)))

function PercentileCells({ p }: { p: Percentiles | null }) {
  return (
    <>
      <TableCell className="text-right num text-xs">{ms(p?.p50)}</TableCell>
      <TableCell className="text-right num text-xs">{ms(p?.p95)}</TableCell>
      <TableCell className="text-right num text-xs text-muted-foreground">{ms(p?.p99)}</TableCell>
    </>
  )
}

function RtfList({ models }: { models: ModelStats[] }) {
  const rows = models.filter((m) => m.rtf).sort((a, b) => (b.rtf?.p50 ?? 0) - (a.rtf?.p50 ?? 0))
  const max = Math.max(0.05, ...rows.map((m) => m.rtf?.p95 ?? 0)) * 1.1
  if (rows.length === 0)
    return <Empty title="No real-time factor yet">It is reported once a loaded model has served a request.</Empty>
  return (
    <ul className="grid gap-2.5">
      {rows.map((m) => {
        const p50 = m.rtf?.p50 ?? 0
        const p95 = m.rtf?.p95 ?? 0
        return (
          <li
            key={m.id}
            className="grid grid-cols-[minmax(0,9.5rem)_minmax(0,1fr)_5.5rem] items-center gap-x-3 text-xs"
          >
            <span className="truncate num" title={m.id}>
              {m.id}
            </span>
            <div
              className="relative h-4"
              role="img"
              aria-label={`${m.id}: real-time factor p50 ${fmtRtf(p50)}, p95 ${fmtRtf(p95)}`}
            >
              <div className="absolute inset-y-0 left-0 w-px bg-chart-axis" />
              <div
                className="absolute top-1 left-0 h-2 rounded-r-[4px] bg-chart-1"
                style={{ width: `${(p50 / max) * 100}%` }}
              />
              <div
                className="absolute top-0 h-4 w-0.5 rounded-full bg-foreground"
                style={{ left: `${(p95 / max) * 100}%` }}
              />
            </div>
            <span className="text-right num">
              {fmtRtf(p50)} <span className="text-muted-foreground">· {fmtSpeed(p50)}</span>
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export default function LatencyPage() {
  const liveState = useLive()
  const [frozen, setFrozen] = useState<LiveState | null>(null)
  const [range, setRange] = useState('300')
  const [model, setModel] = useState(ALL)
  const [metric, setMetric] = useState<Metric>('first_partial')
  const [selected, setSelected] = useState<RequestTrace | null>(null)
  const targetMs = useStore(prefs).targetMs
  const data = frozen ?? liveState
  const paused = frozen != null
  const missed = paused ? liveState.requests.filter((r) => !frozen.requests.some((f) => f.id === r.id)).length : 0

  const now = data.lastEventAt ?? data.history[data.history.length - 1]?.t ?? null
  const rangeMs = Number(range) * 1000
  const modelId = model === ALL ? null : model

  const view = useMemo(() => {
    if (now == null) return null
    const to = now + 500
    const from = to - rangeMs
    const traces = filterTraces(data.requests, { model: modelId, from, to })
    const ok = traces.filter((t) => t.status === 'ok')
    const vals = (m: Metric) => ok.map((t) => metricValue(t, m)).filter((v): v is number => v != null)
    const buckets = 30
    const series = bucketize(traces, metric, from, to, buckets)
    const spark = (m: Metric, key: 'p50' | 'p95') => bucketize(traces, m, from, to, 20).map((b) => b[key])
    const rtfs = ok.map((t) => t.rtf).filter((v): v is number => v != null)
    const hist = data.history.filter((h) => h.t >= from)
    const metricVals = vals(metric)
    const cap = metricVals.length ? percentile(metricVals, 99) * 1.05 : undefined
    return {
      from,
      to,
      traces,
      series,
      fp: summarize(vals('first_partial')),
      ttfa: summarize(vals('ttfa')),
      rtfP50: rtfs.length ? percentile(rtfs, 50) : null,
      sparkFp50: spark('first_partial', 'p50'),
      sparkFp95: spark('first_partial', 'p95'),
      sparkTtfa: spark('ttfa', 'p95'),
      metricSummary: summarize(metricVals),
      bins: histogram(metricVals, 24, cap),
      hist,
      errors: traces.length - ok.length,
    }
  }, [data.requests, data.history, now, rangeMs, modelId, metric])

  const stats = data.stats
  const loadedModels = (stats?.models ?? []).filter((m) => m.task !== 'vad')
  const shownModels = modelId ? loadedModels.filter((m) => m.id === modelId) : loadedModels
  const loading = stats == null && data.status !== 'open'
  const kinds = new Set(view?.traces.map((t) => t.kind))
  const waterfallKind = kinds.size > 1 ? 'mixed' : kinds.has('tts') ? 'tts' : 'stt'
  const rows = view?.traces.slice(0, 40) ?? []
  const seriesPoints = view?.series.filter((p) => p.p50 != null).length ?? 0
  const seriesHasData = seriesPoints > 0
  // With only a handful of buckets a line is nearly invisible, so mark the points.
  const sparse = seriesPoints <= 3

  const toolbar = (
    <>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={range}
        onValueChange={(v) => v && setRange(v)}
        aria-label="Time range"
      >
        {RANGES.map((r) => (
          <ToggleGroupItem key={r.value} value={r.value} className="px-2.5 text-xs">
            {r.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <Select value={model} onValueChange={setModel}>
        <SelectTrigger className="h-8 w-52 max-sm:min-w-44 max-sm:flex-1" aria-label="Model">
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
          <SelectItem value={ALL}>All models</SelectItem>
          {loadedModels.map((m) => (
            <SelectItem key={m.id} value={m.id}>
              <span className="num text-[13px]">{m.id}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={metric} onValueChange={(v) => setMetric(v as Metric)}>
        <SelectTrigger className="h-8 w-48 max-sm:min-w-40 max-sm:flex-1" aria-label="Latency metric">
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
          {(Object.keys(METRIC_LABEL) as Metric[]).map((m) => (
            <SelectItem key={m} value={m}>
              {METRIC_LABEL[m]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div className="ml-auto flex items-center gap-2">
        <span className="text-xs text-muted-foreground" aria-live="polite">
          {paused ? (
            <>
              Paused at <span className="num">{now ? fmtClock(now) : '–'}</span>
              {missed > 0 && (
                <>
                  {' '}
                  · <span className="num text-foreground">{missed}</span> new
                </>
              )}
            </>
          ) : data.status === 'open' ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="size-1.5 live-dot rounded-full bg-good" aria-hidden /> Live
            </span>
          ) : (
            'Waiting for the feed'
          )}
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setFrozen(paused ? null : liveState)}
          aria-pressed={paused}
          data-testid="pause"
        >
          {paused ? <Play className="fill-current" /> : <Pause className="fill-current" />}
          {paused ? 'Resume' : 'Pause'}
        </Button>
      </div>
    </>
  )

  const fpState = budgetState(view?.fp.p95, targetMs)
  const ttfaState = budgetState(view?.ttfa.p95, targetMs)
  const histTicks = view?.bins.length
    ? view.bins.filter((_, i) => i % Math.ceil(view.bins.length / 6) === 0).map((b) => b.x0)
    : []

  const seriesMax = Math.max(targetMs * 1.15, ...(view?.series ?? []).map((p) => p.p95 ?? 0))
  const yStep = niceStep(seriesMax, 4)
  const yMax = Math.ceil(seriesMax / yStep) * yStep
  const yTicks = Array.from({ length: Math.round(yMax / yStep) + 1 }, (_, i) => i * yStep)
  // The target line sits on the lower edge of the bin that contains it.
  const targetBin = view?.bins.find((b) => targetMs >= b.x0 && targetMs < b.x1)?.x0 ?? null

  return (
    <Page title="Latency" toolbar={toolbar}>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatTile
          loading={loading}
          label="Active streams"
          value={fmtInt(stats?.active_streams)}
          context={`${fmtInt(stats?.system.threads)} threads`}
          trend={view?.hist.map((h) => h.stats.active_streams)}
        />
        <StatTile
          loading={loading}
          label="Queue depth"
          value={fmtInt(stats?.queue_depth)}
          status={stats ? (stats.queue_depth === 0 ? 'good' : stats.queue_depth < 5 ? 'warning' : 'bad') : undefined}
          context={stats ? (stats.queue_depth === 0 ? 'nothing waiting' : 'requests waiting') : ''}
          trend={view?.hist.map((h) => h.stats.queue_depth)}
        />
        <StatTile
          loading={loading}
          label="First partial p50"
          value={ms(view?.fp.p50)}
          unit="ms"
          context={`${fmtInt(view?.fp.n)} requests`}
          trend={view?.sparkFp50}
        />
        <StatTile
          loading={loading}
          label="First partial p95"
          value={ms(view?.fp.p95)}
          unit="ms"
          status={fpState}
          context={`target ${targetMs} ms`}
          trend={view?.sparkFp95}
          trendColor="var(--chart-2)"
        />
        <StatTile
          loading={loading}
          label="First audio p95"
          value={ms(view?.ttfa.p95)}
          unit="ms"
          status={ttfaState}
          context={`${fmtInt(view?.ttfa.n)} TTS requests`}
          trend={view?.sparkTtfa}
          trendColor="var(--chart-2)"
        />
        <StatTile
          loading={loading}
          label="Real-time factor p50"
          value={fmtRtf(view?.rtfP50)}
          context={view?.rtfP50 ? `${fmtSpeed(view.rtfP50)} real time` : 'lower is faster'}
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <ChartPanel
          className="xl:col-span-2"
          title={`${METRIC_LABEL[metric]} over time`}
          description={`Percentiles per ${Math.round(rangeMs / 30000)} s bucket · ms`}
          legend={
            <>
              <LegendKey color="var(--chart-1)" label="p50" />
              <LegendKey color="var(--chart-2)" label="p95" />
              <LegendKey color="var(--foreground)" label={`target ${targetMs} ms`} dashed />
            </>
          }
          chart={
            loading ? (
              <Skeleton className="h-60 w-full" />
            ) : !seriesHasData ? (
              <Empty icon={Activity} title="No requests in this window" className="h-60">
                Nothing with a {METRIC_LABEL[metric].toLowerCase()} measurement completed in the selected range.
              </Empty>
            ) : (
              <ChartContainer
                config={lineConfig}
                className={cn('aspect-auto h-60 w-full', paused && 'opacity-80')}
                data-testid="latency-series"
              >
                <LineChart data={view?.series} margin={{ top: 8, right: 28, bottom: 0, left: 0 }} accessibilityLayer>
                  <CartesianGrid {...gridProps} />
                  <XAxis
                    dataKey="t"
                    type="number"
                    scale="time"
                    domain={[view?.from ?? 0, view?.to ?? 1]}
                    tickFormatter={(t: number) => fmtClock(t)}
                    {...axisProps}
                    minTickGap={48}
                  />
                  <YAxis
                    {...axisProps}
                    axisLine={false}
                    width={44}
                    tickFormatter={(v: number) => String(v)}
                    domain={[0, yMax]}
                    ticks={yTicks}
                  />
                  <ChartTooltip
                    cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
                    isAnimationActive={false}
                    content={({ active, payload }) => {
                      const p = payload?.[0]?.payload as
                        (typeof view extends null ? never : NonNullable<typeof view>['series'][number]) | undefined
                      if (!active || !p) return null
                      return (
                        <TipBox
                          title={fmtClock(p.t)}
                          rows={[
                            { color: 'var(--chart-2)', label: 'p95', value: fmtMs(p.p95) },
                            { color: 'var(--chart-1)', label: 'p50', value: fmtMs(p.p50) },
                          ]}
                          footer={`${p.n} request${p.n === 1 ? '' : 's'}`}
                        />
                      )
                    }}
                  />
                  <ReferenceLine y={targetMs} stroke="var(--foreground)" strokeOpacity={0.45} strokeWidth={1} />
                  <Line
                    dataKey="p95"
                    stroke="var(--chart-2)"
                    strokeWidth={2}
                    dot={sparse ? { r: 4, fill: 'var(--chart-2)', stroke: 'var(--card)', strokeWidth: 2 } : false}
                    activeDot={{ r: 4, stroke: 'var(--card)', strokeWidth: 2 }}
                    connectNulls
                    isAnimationActive={false}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  <Line
                    dataKey="p50"
                    stroke="var(--chart-1)"
                    strokeWidth={2}
                    dot={sparse ? { r: 4, fill: 'var(--chart-1)', stroke: 'var(--card)', strokeWidth: 2 } : false}
                    activeDot={{ r: 4, stroke: 'var(--card)', strokeWidth: 2 }}
                    connectNulls
                    isAnimationActive={false}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                </LineChart>
              </ChartContainer>
            )
          }
          table={
            <DataTable
              caption={`${METRIC_LABEL[metric]} percentiles over time`}
              head={['Bucket start', 'Requests', 'p50 ms', 'p95 ms']}
              rows={(view?.series ?? []).filter((p) => p.n > 0).map((p) => [fmtClock(p.t), p.n, ms(p.p50), ms(p.p95)])}
            />
          }
        />

        <ChartPanel
          title={`${METRIC_LABEL[metric]} distribution`}
          description={modelId ? <span className="num">{modelId}</span> : 'All models · requests per latency bin'}
          legend={
            <>
              <LegendKey color="var(--chart-1)" shape="rect" label="requests" />
              {targetBin != null && <LegendKey color="var(--foreground)" label="target" dashed />}
              <span className="ml-auto num">
                p50 {ms(view?.metricSummary.p50)} · p95 {ms(view?.metricSummary.p95)} ms
              </span>
            </>
          }
          chart={
            loading ? (
              <Skeleton className="h-60 w-full" />
            ) : !view?.bins.length ? (
              <Empty icon={Activity} title="Nothing to bin yet" className="h-60">
                The histogram fills in as requests complete.
              </Empty>
            ) : (
              <ChartContainer config={histConfig} className="aspect-auto h-60 w-full" data-testid="latency-histogram">
                <BarChart
                  data={view.bins}
                  margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
                  barCategoryGap={2}
                  accessibilityLayer
                >
                  <CartesianGrid {...gridProps} />
                  <XAxis
                    dataKey="x0"
                    ticks={histTicks}
                    tickFormatter={(v: number) => String(Math.round(v))}
                    {...axisProps}
                    interval={0}
                  />
                  <YAxis {...axisProps} axisLine={false} width={32} allowDecimals={false} />
                  <ChartTooltip
                    cursor={{ fill: 'var(--muted)', opacity: 0.5 }}
                    isAnimationActive={false}
                    content={({ active, payload }) => {
                      const b = payload?.[0]?.payload as { x0: number; x1: number; count: number } | undefined
                      if (!active || !b) return null
                      return (
                        <TipBox
                          title={`${Math.round(b.x0)}–${Math.round(b.x1)} ms`}
                          rows={[{ color: 'var(--chart-1)', label: 'requests', value: String(b.count) }]}
                        />
                      )
                    }}
                  />
                  <Bar
                    dataKey="count"
                    fill="var(--chart-1)"
                    radius={[3, 3, 0, 0]}
                    maxBarSize={24}
                    isAnimationActive={false}
                  />
                  {targetBin != null && (
                    <ReferenceLine
                      x={targetBin}
                      position="start"
                      stroke="var(--foreground)"
                      strokeOpacity={0.45}
                      strokeWidth={1}
                    />
                  )}
                </BarChart>
              </ChartContainer>
            )
          }
          table={
            <DataTable
              caption={`${METRIC_LABEL[metric]} histogram`}
              head={['Bin (ms)', 'Requests']}
              rows={(view?.bins ?? []).map((b) => [`${Math.round(b.x0)}–${Math.round(b.x1)}`, b.count])}
            />
          }
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Panel
          title="Request waterfall"
          description={`Newest first · ${rows.length} of ${fmtInt(view?.traces.length)} in range${view?.errors ? ` · ${view.errors} failed` : ''} · click a row for the breakdown`}
          className="xl:col-span-2"
          bodyClassName="max-h-[34rem] overflow-y-auto"
        >
          {loading ? (
            <div className="grid gap-2">
              {Array.from({ length: 8 }, (_, i) => (
                <Skeleton key={i} className="h-5" />
              ))}
            </div>
          ) : (
            <Waterfall traces={rows} onSelect={setSelected} selectedId={selected?.id ?? null} kind={waterfallKind} />
          )}
        </Panel>

        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Real-time factor" description="Processing time ÷ audio time, per model · bar p50, tick p95">
            {loading ? <Skeleton className="h-28 w-full" /> : <RtfList models={shownModels} />}
          </Panel>
          {isDemo ? (
            <p className="rounded-xl border border-dashed border-border px-4 py-3 text-xs leading-5 text-muted-foreground">
              CPU and memory are not shown here: a web page cannot measure its own. On a bolo server these tiles are
              live.
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <StatTile
                loading={loading}
                label="CPU"
                value={stats ? Math.round(stats.system.cpu_percent) : '–'}
                unit="%"
                context="of one core"
                trend={view?.hist.map((h) => h.stats.system.cpu_percent)}
              />
              <StatTile
                loading={loading}
                label="Memory"
                value={stats ? fmtBytes(stats.system.rss_bytes).split(' ')[0] : '–'}
                unit={stats ? fmtBytes(stats.system.rss_bytes).split(' ')[1] : undefined}
                context="resident"
                trend={view?.hist.map((h) => h.stats.system.rss_bytes)}
              />
            </div>
          )}
          <Panel title="Since start" bodyClassName="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
            <div>
              <div className="text-muted-foreground">Requests</div>
              <div className="mt-0.5 num text-sm font-medium">{fmtInt(stats?.totals.requests)}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Errors</div>
              <div className="mt-0.5 num text-sm font-medium">
                {fmtInt(stats?.totals.errors)}
                {stats && stats.totals.requests > 0 && (
                  <span className="font-normal text-muted-foreground">
                    {' '}
                    · {((stats.totals.errors / stats.totals.requests) * 100).toFixed(2)}%
                  </span>
                )}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">Audio transcribed</div>
              <div className="mt-0.5 num text-sm font-medium">{fmtAudioTotal(stats?.totals.audio_seconds_in)}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Audio synthesised</div>
              <div className="mt-0.5 num text-sm font-medium">{fmtAudioTotal(stats?.totals.audio_seconds_out)}</div>
            </div>
          </Panel>
        </div>
      </div>

      <Panel
        title="Per model"
        description={
          isDemo
            ? 'Measured in this tab over the last five minutes · ms'
            : 'Server-reported percentiles over its own window · ms'
        }
        className="mt-4"
        bodyClassName="p-0"
      >
        {loading ? (
          <div className="p-4">
            <Skeleton className="h-24 w-full" />
          </div>
        ) : shownModels.length === 0 ? (
          <Empty title={isDemo ? 'Nothing has run yet' : 'No models are loaded'}>
            {isDemo
              ? 'This dashboard shows only what you run in this tab. Transcribe something in the playground and come back.'
              : 'Load a model from the Models screen to see its latency here.'}
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <Table className="min-w-[900px]">
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead rowSpan={2} className="pl-4 align-bottom">
                    Model
                  </TableHead>
                  <TableHead rowSpan={2} className="text-right align-bottom">
                    Active
                  </TableHead>
                  <TableHead rowSpan={2} className="text-right align-bottom">
                    Requests
                  </TableHead>
                  <TableHead colSpan={3} className="h-7 border-l border-border text-center text-[11px]">
                    First partial
                  </TableHead>
                  <TableHead colSpan={3} className="h-7 border-l border-border text-center text-[11px]">
                    Final
                  </TableHead>
                  <TableHead colSpan={3} className="h-7 border-l border-border text-center text-[11px]">
                    First audio
                  </TableHead>
                  <TableHead rowSpan={2} className="border-l border-border text-right align-bottom">
                    RTF p50
                  </TableHead>
                  <TableHead rowSpan={2} className="pr-4 text-right align-bottom">
                    Memory
                  </TableHead>
                </TableRow>
                <TableRow className="hover:bg-transparent">
                  {[0, 1, 2].flatMap((g) =>
                    ['p50', 'p95', 'p99'].map((h, i) => (
                      <TableHead
                        key={`${g}-${h}`}
                        className={cn('h-7 text-right text-[11px]', i === 0 && 'border-l border-border')}
                      >
                        {h}
                      </TableHead>
                    )),
                  )}
                </TableRow>
              </TableHeader>
              <TableBody>
                {shownModels.map((m) => (
                  <TableRow key={m.id}>
                    <TableCell className="pl-4">
                      <span className="num text-[13px]">{m.id}</span>
                      <span className="ml-2 text-[10px] font-medium text-muted-foreground uppercase">{m.task}</span>
                    </TableCell>
                    <TableCell className="text-right num text-xs">{m.active}</TableCell>
                    <TableCell className="text-right num text-xs">{fmtInt(m.requests)}</TableCell>
                    <PercentileCells p={m.latency_ms.first_partial} />
                    <PercentileCells p={m.latency_ms.final} />
                    <PercentileCells p={m.latency_ms.ttfa} />
                    <TableCell className="text-right num text-xs">{fmtRtf(m.rtf?.p50)}</TableCell>
                    <TableCell className="pr-4 text-right num text-xs">{fmtBytes(m.memory_bytes)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>

      <RequestDrawer trace={selected} onClose={() => setSelected(null)} />
    </Page>
  )
}
