import { Download, Gauge, Play, Square, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from 'recharts'
import { toast } from 'sonner'
import { axisProps, ChartPanel, DataTable, gridProps, LegendKey, TipBox } from '@/components/charts/chart-kit'
import { Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { Empty, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { describeError, pickDefault, useModels, useSamples } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { SERVER_ONLY_HINT } from '@/components/demo'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { BenchPoint } from '@/lib/api/types'
import {
  BENCH_METRIC_LABEL,
  ceilingFor,
  defaultMetric,
  download,
  loadRuns,
  parseLevels,
  saveRuns,
  toCsv,
  toJson,
  type BenchMetric,
  type BenchRun,
} from '@/lib/bench'
import { fmtClock, fmtMs, fmtRtf } from '@/lib/format'
import { niceStep } from '@/lib/latency/stats'
import { prefs, setPref } from '@/lib/prefs'

class BenchFailed extends Error {}

const PRESETS = isDemo
  ? ['1, 2, 4', '1, 2, 4, 8', '1, 2, 3, 4, 6, 8']
  : ['1, 2, 4, 8', '1, 2, 4, 8, 16', '1, 2, 3, 4, 6, 8, 12, 16']
const PRESET_LABELS = isDemo ? ['up to 4', 'up to 8', 'fine'] : ['up to 8', 'up to 16', 'fine']
const NONE = '__none'

const latencyConfig = {
  p50: { label: 'p50', color: 'var(--chart-1)' },
  p95: { label: 'p95', color: 'var(--chart-2)' },
} satisfies ChartConfig
const dot = (color: string) => ({ r: 4, fill: color, stroke: 'var(--card)', strokeWidth: 2 })

interface Row {
  c: number
  p50: number | null
  p95: number | null
  thr: number | null
  bp95: number | null
  bthr: number | null
}

function rowsFor(a: BenchPoint[], b: BenchPoint[] | null, metric: BenchMetric): Row[] {
  const levels = [...new Set([...a.map((p) => p.concurrency), ...(b ?? []).map((p) => p.concurrency)])].sort(
    (x, y) => x - y,
  )
  return levels.map((c) => {
    const pa = a.find((p) => p.concurrency === c)
    const pb = b?.find((p) => p.concurrency === c)
    return {
      c,
      p50: pa?.[metric]?.p50 ?? null,
      p95: pa?.[metric]?.p95 ?? null,
      thr: pa?.throughput_x_realtime ?? null,
      bp95: pb?.[metric]?.p95 ?? null,
      bthr: pb?.throughput_x_realtime ?? null,
    }
  })
}

function niceMax(v: number, ticks = 4): { max: number; ticks: number[] } {
  const step = niceStep(Math.max(v, 1), ticks)
  const max = Math.ceil(v / step) * step
  return { max, ticks: Array.from({ length: Math.round(max / step) + 1 }, (_, i) => i * step) }
}

function runLabel(r: BenchRun): string {
  return `${r.request.model} · ${fmtClock(r.startedAt)}`
}

export default function BenchmarkPage() {
  const [kind, setKind] = useState<'stt' | 'tts'>('stt')
  const { models, isLoading } = useModels(kind)
  const samples = useSamples()
  const p = useStore(prefs)
  const [modelId, setModelId] = useState<string | null>(null)
  const [sample, setSample] = useState<string | null>(null)
  const [levelsText, setLevelsText] = useState(PRESETS[0] as string)
  const [duration, setDuration] = useState(isDemo ? 6 : 10)
  const [progress, setProgress] = useState(0)
  const [pending, setPending] = useState<BenchRun | null>(null)
  const [runs, setRuns] = useState<BenchRun[]>(loadRuns)
  const [currentId, setCurrentId] = useState<string | null>(() => loadRuns()[0]?.id ?? null)
  const [compareId, setCompareId] = useState<string>(NONE)
  const [metricChoice, setMetricChoice] = useState<BenchMetric | null>(null)

  const model = useMemo(
    () => pickDefault(models, modelId ?? (kind === 'stt' ? p.sttModel : p.ttsModel)),
    [models, modelId, kind, p.sttModel, p.ttsModel],
  )
  const sampleOptions = (samples.data?.samples ?? []).filter((s) => s.kind === (kind === 'stt' ? 'audio' : 'text'))
  const sampleId = sampleOptions.find((s) => s.id === sample)?.id ?? sampleOptions[0]?.id ?? null
  const levels = parseLevels(levelsText)

  const alive = useRef(true)
  useEffect(
    () => () => {
      alive.current = false
    },
    [],
  )

  const start = async () => {
    if (!model || !sampleId || levels.length === 0) return
    const request = { kind, model: model.id, concurrency: levels, duration_s: duration, sample: `builtin:${sampleId}` }
    let run: BenchRun
    let jobId: string
    try {
      jobId = (await api.bench(request)).job_id
      run = { id: `${jobId}:${Date.now()}`, startedAt: Date.now(), request, points: [] }
      setPending(run)
      setProgress(0)
      setMetricChoice(null)
    } catch (err) {
      toast.error('Could not start the benchmark', { description: describeError(err) })
      return
    }
    // Poll the job; partial results arrive after every level.
    let failures = 0
    while (alive.current) {
      await new Promise((r) => setTimeout(r, 400))
      try {
        const j = await api.benchJob(jobId)
        failures = 0
        if (!alive.current) return
        run = { ...run, points: j.result?.points ?? run.points }
        setPending(run)
        setProgress(j.progress ?? 0)
        if (j.state === 'done') {
          const done = run
          setRuns((prev) => {
            const next = [done, ...prev.filter((r) => r.id !== done.id)].slice(0, 12)
            saveRuns(next)
            return next
          })
          setCurrentId(done.id)
          setPending(null)
          toast.success('Benchmark finished', { description: `${done.points.length} concurrency levels measured` })
          return
        }
        if (j.state === 'error') throw new BenchFailed(j.error ?? 'The benchmark job failed')
      } catch (err) {
        if (err instanceof BenchFailed || ++failures >= 5) {
          if (alive.current) {
            setPending(null)
            toast.error('Benchmark failed', { description: describeError(err) })
          }
          return
        }
      }
    }
  }

  const running = pending != null
  const current = running ? pending : (runs.find((r) => r.id === currentId) ?? null)
  const compare = !running && compareId !== NONE ? (runs.find((r) => r.id === compareId) ?? null) : null
  const runKind = (current?.request.kind ?? kind) as 'stt' | 'tts'
  const metric: BenchMetric =
    metricChoice && (runKind === 'tts') === (metricChoice === 'ttfa_ms') ? metricChoice : defaultMetric(runKind)
  const rows = current ? rowsFor(current.points, compare?.points ?? null, metric) : []
  const ceiling = current && current.points.length ? ceilingFor(current.points, metric, p.targetMs) : null
  const ceilingB = compare ? ceilingFor(compare.points, metric, p.targetMs) : null
  const latY = niceMax(Math.max(p.targetMs * 1.2, ...rows.flatMap((r) => [r.p95 ?? 0, r.bp95 ?? 0])))
  const thrY = niceMax(Math.max(1, ...rows.flatMap((r) => [r.thr ?? 0, r.bthr ?? 0])))
  const firstMiss = current?.points.length && ceiling == null
  const allPass = ceiling != null && ceiling === Math.max(...(current?.points.map((x) => x.concurrency) ?? [0]))

  const exportRun = (fmt: 'json' | 'csv') => {
    if (!current) return
    const name = `bolo-bench-${current.request.model}-${new Date(current.startedAt).toISOString().slice(0, 19).replace(/[:T]/g, '')}`
    if (fmt === 'json') download(`${name}.json`, toJson(current), 'application/json')
    else download(`${name}.csv`, toCsv(current), 'text/csv')
  }

  const removeRun = (id: string) => {
    const next = runs.filter((r) => r.id !== id)
    setRuns(next)
    saveRuns(next)
    if (currentId === id) setCurrentId(next[0]?.id ?? null)
    if (compareId === id) setCompareId(NONE)
  }

  const comparing = compare != null
  const seriesA = comparing ? (current ? runLabel(current) : 'A') : 'p95'
  const xAxis = <XAxis dataKey="c" type="category" {...axisProps} />

  return (
    <Page title="Benchmark">
      {isDemo && (
        <p
          className="mb-4 rounded-xl border border-border bg-card px-4 py-3 text-[13px] leading-5 text-muted-foreground"
          role="note"
          data-testid="bench-demo-note"
        >
          <span className="font-medium text-foreground">This sweep runs in your browser.</span> The page has one
          recogniser, so "concurrency" here means how many utterances are handed to it at once; they are decoded one
          after another and the wait is queueing. A bolo server batches and runs several workers, so its curve looks
          different. Use these numbers to compare devices and browsers, not to size a server.
        </p>
      )}
      <div className="grid gap-4 xl:grid-cols-[20rem_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="Sweep"
            description={
              isDemo ? 'Runs in this tab against a built-in sample' : 'Runs on the server against a built-in sample'
            }
          >
            <form
              className="grid gap-4"
              onSubmit={(e) => {
                e.preventDefault()
                void start()
              }}
            >
              <div className="grid gap-1.5">
                <Label className="text-xs" id="bench-kind">
                  Task
                </Label>
                <ToggleGroup
                  type="single"
                  variant="outline"
                  size="sm"
                  value={kind}
                  onValueChange={(v) => v && (setKind(v as 'stt' | 'tts'), setModelId(null), setSample(null))}
                  aria-labelledby="bench-kind"
                  disabled={running}
                  className="w-full"
                >
                  <ToggleGroupItem value="stt" className="flex-1 text-xs">
                    Speech to text
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="tts"
                    className="flex-1 text-xs"
                    disabled={isDemo}
                    title={isDemo ? SERVER_ONLY_HINT : undefined}
                  >
                    Text to speech
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="bench-model" className="text-xs">
                  Model
                </Label>
                <ModelPicker
                  id="bench-model"
                  models={models}
                  value={model?.id ?? null}
                  onChange={setModelId}
                  loading={isLoading}
                  disabled={running}
                  className="w-full"
                />
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="bench-sample" className="text-xs">
                  Sample
                </Label>
                <Select
                  value={sampleId ?? undefined}
                  onValueChange={setSample}
                  disabled={running || sampleOptions.length === 0}
                >
                  <SelectTrigger id="bench-sample" className="h-8 w-full">
                    <SelectValue placeholder={samples.isLoading ? 'Loading…' : 'No samples'} />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start">
                    {sampleOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        <span className="num text-[13px]">{s.id}</span>
                        {s.seconds != null && (
                          <span className="ml-2 num text-xs text-muted-foreground">{s.seconds.toFixed(1)} s</span>
                        )}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="bench-levels" className="text-xs">
                  Concurrency levels
                </Label>
                <Input
                  id="bench-levels"
                  value={levelsText}
                  onChange={(e) => setLevelsText(e.target.value)}
                  className="h-8 num"
                  disabled={running}
                  aria-describedby="bench-levels-hint"
                  aria-invalid={levels.length === 0}
                  spellCheck={false}
                />
                <div className="flex flex-wrap gap-1">
                  {PRESETS.map((preset, i) => (
                    <Button
                      key={preset}
                      type="button"
                      size="xs"
                      variant={levelsText === preset ? 'secondary' : 'ghost'}
                      className="num"
                      onClick={() => setLevelsText(preset)}
                      disabled={running}
                    >
                      {PRESET_LABELS[i]}
                    </Button>
                  ))}
                </div>
                <p id="bench-levels-hint" className="text-[11px] leading-4 text-muted-foreground">
                  {levels.length === 0
                    ? 'Enter numbers or ranges, e.g. 1, 2, 4-8.'
                    : `${levels.length} level${levels.length === 1 ? '' : 's'} · about ${levels.length * duration} s in total`}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label htmlFor="bench-duration" className="text-xs">
                    Seconds per level
                  </Label>
                  <Input
                    id="bench-duration"
                    type="number"
                    min={2}
                    max={120}
                    value={duration}
                    onChange={(e) => setDuration(Math.max(2, Math.min(120, Number(e.target.value) || 10)))}
                    className="h-8 num"
                    disabled={running}
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="bench-target" className="text-xs">
                    Target p95 (ms)
                  </Label>
                  <Input
                    id="bench-target"
                    type="number"
                    min={50}
                    max={5000}
                    step={50}
                    value={p.targetMs}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      if (v >= 50 && v <= 5000) setPref('targetMs', Math.round(v))
                    }}
                    className="h-8 num"
                  />
                </div>
              </div>
              {running ? (
                <Button type="button" variant="outline" disabled>
                  <Square className="fill-current" /> Running…
                </Button>
              ) : (
                <Button
                  type="submit"
                  disabled={!model?.installed || !sampleId || levels.length === 0}
                  data-testid="bench-run"
                >
                  <Play className="fill-current" /> Run benchmark
                </Button>
              )}
              {model && !model.installed && (
                <p className="text-xs text-warning-text">This model is not installed. Pull it first.</p>
              )}
            </form>
          </Panel>

          <Panel title="Runs" description="Kept in this browser" bodyClassName="p-0">
            {runs.length === 0 ? (
              <p className="px-4 py-5 text-[13px] leading-5 text-muted-foreground">
                Finished runs are listed here so you can reopen and compare them.
              </p>
            ) : (
              <ul>
                {runs.map((r) => (
                  <li key={r.id} className="flex items-center gap-1 border-b border-border px-2 last:border-b-0">
                    <button
                      type="button"
                      onClick={() => setCurrentId(r.id)}
                      aria-current={current?.id === r.id}
                      className="min-w-0 flex-1 rounded-md px-2 py-2 text-left hover:bg-muted/50 aria-[current=true]:bg-muted/60"
                    >
                      <div className="truncate num text-[13px]">{r.request.model}</div>
                      <div className="num text-[11px] text-muted-foreground">
                        {fmtClock(r.startedAt)} · {r.points.length} levels · up to{' '}
                        {Math.max(0, ...r.points.map((x) => x.concurrency))}
                      </div>
                    </button>
                    <Button
                      size="icon-sm"
                      variant="ghost"
                      aria-label={`Delete run ${runLabel(r)}`}
                      onClick={() => removeRun(r.id)}
                    >
                      <Trash2 />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          {running && (
            <div className="rounded-xl border border-border bg-card px-4 py-3" role="status">
              <div className="mb-2 flex items-baseline justify-between gap-3 text-sm">
                <span>
                  Benchmarking <span className="num font-medium">{pending.request.model}</span>
                </span>
                <span className="num text-xs text-muted-foreground">
                  level {Math.min(pending.request.concurrency.length, pending.points.length + 1)} of{' '}
                  {pending.request.concurrency.length} · {Math.round(progress * 100)}%
                </span>
              </div>
              <Progress value={progress * 100} className="h-1.5" aria-label="Benchmark progress" />
            </div>
          )}

          {!current ? (
            <div className="rounded-xl border border-dashed border-border bg-card">
              <Empty icon={Gauge} title="No benchmark yet" className="py-20">
                Pick a model and run a sweep. You will get latency and throughput at each concurrency level, and the
                highest level that still meets your target.
              </Empty>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <div className="mr-auto min-w-0">
                  <div className="truncate num text-sm font-medium">{current.request.model}</div>
                  <div className="num text-xs text-muted-foreground">
                    {current.request.sample} · {current.request.duration_s} s per level · {fmtClock(current.startedAt)}
                  </div>
                </div>
                {runKind === 'stt' && (
                  <Select value={metric} onValueChange={(v) => setMetricChoice(v as BenchMetric)}>
                    <SelectTrigger className="h-8 w-40" aria-label="Latency metric">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent position="popper" align="end">
                      <SelectItem value="final_ms">Final</SelectItem>
                      <SelectItem value="first_partial_ms">First partial</SelectItem>
                    </SelectContent>
                  </Select>
                )}
                <Select
                  value={compare ? compareId : NONE}
                  onValueChange={setCompareId}
                  disabled={running || runs.length < 2}
                >
                  <SelectTrigger className="h-8 w-56" aria-label="Compare with another run">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" align="end">
                    <SelectItem value={NONE}>
                      {runs.length < 2 ? 'Compare: needs two runs' : 'Compare with…'}
                    </SelectItem>
                    {runs
                      .filter((r) => r.id !== current.id)
                      .map((r) => (
                        <SelectItem key={r.id} value={r.id}>
                          <span className="num text-[13px]">{runLabel(r)}</span>
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button size="sm" variant="outline" className="h-8" disabled={current.points.length === 0}>
                      <Download /> Export
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onSelect={() => exportRun('json')}>JSON</DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => exportRun('csv')}>CSV</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="rounded-xl border border-border bg-card px-5 py-4" data-testid="ceiling">
                <div className="text-xs text-muted-foreground">
                  Concurrency ceiling · {BENCH_METRIC_LABEL[metric].toLowerCase()} p95 within{' '}
                  <span className="num">{p.targetMs} ms</span>
                </div>
                <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="text-5xl leading-none font-semibold tracking-tight">
                    {current.points.length === 0 ? '–' : (ceiling ?? 0)}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {current.points.length === 0
                      ? 'measuring…'
                      : firstMiss
                        ? 'Even the lowest level misses the target.'
                        : allPass
                          ? 'streams, and every level tested met the target. Try higher levels to find the limit.'
                          : `concurrent stream${ceiling === 1 ? '' : 's'} before p95 crosses the target`}
                  </span>
                  {compare && (
                    <span className="ml-auto num text-sm text-muted-foreground">
                      vs {ceilingB ?? 0} for {compare.request.model}
                    </span>
                  )}
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <ChartPanel
                  title={`${BENCH_METRIC_LABEL[metric]} vs concurrency`}
                  description="ms, by concurrent streams · lower is better"
                  legend={
                    comparing ? (
                      <>
                        <LegendKey color="var(--chart-1)" label={<span className="num">{seriesA} p95</span>} />
                        <LegendKey
                          color="var(--chart-2)"
                          label={<span className="num">{runLabel(compare)} p95</span>}
                        />
                        <LegendKey color="var(--foreground)" label="target" dashed />
                      </>
                    ) : (
                      <>
                        <LegendKey color="var(--chart-1)" label="p50" />
                        <LegendKey color="var(--chart-2)" label="p95" />
                        <LegendKey color="var(--foreground)" label={`target ${p.targetMs} ms`} dashed />
                      </>
                    )
                  }
                  chart={
                    <ChartContainer
                      config={latencyConfig}
                      className="aspect-auto h-64 w-full"
                      data-testid="bench-latency"
                    >
                      <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 0, left: 0 }} accessibilityLayer>
                        <CartesianGrid {...gridProps} />
                        {xAxis}
                        <YAxis {...axisProps} axisLine={false} width={48} domain={[0, latY.max]} ticks={latY.ticks} />
                        <ChartTooltip
                          cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
                          isAnimationActive={false}
                          content={({ active, payload }) => {
                            const r = payload?.[0]?.payload as Row | undefined
                            if (!active || !r) return null
                            return (
                              <TipBox
                                title={`${r.c} concurrent`}
                                rows={
                                  comparing
                                    ? [
                                        { color: 'var(--chart-1)', label: 'this run p95', value: fmtMs(r.p95) },
                                        { color: 'var(--chart-2)', label: 'other run p95', value: fmtMs(r.bp95) },
                                      ]
                                    : [
                                        { color: 'var(--chart-2)', label: 'p95', value: fmtMs(r.p95) },
                                        { color: 'var(--chart-1)', label: 'p50', value: fmtMs(r.p50) },
                                      ]
                                }
                                footer={
                                  r.p95 != null
                                    ? r.p95 <= p.targetMs
                                      ? 'within target'
                                      : `over target by ${fmtMs(r.p95 - p.targetMs)}`
                                    : undefined
                                }
                              />
                            )
                          }}
                        />
                        <ReferenceLine y={p.targetMs} stroke="var(--foreground)" strokeOpacity={0.45} strokeWidth={1} />
                        {ceiling != null && !allPass && (
                          <ReferenceLine
                            x={ceiling}
                            stroke="var(--foreground)"
                            strokeOpacity={0.45}
                            strokeWidth={1}
                            label={{
                              value: `ceiling ${ceiling}`,
                              position: 'insideTopLeft',
                              fill: 'var(--foreground)',
                              fontSize: 11,
                            }}
                          />
                        )}
                        {comparing ? (
                          <>
                            <Line
                              dataKey="bp95"
                              stroke="var(--chart-2)"
                              strokeWidth={2}
                              dot={dot('var(--chart-2)')}
                              isAnimationActive={false}
                              connectNulls
                            />
                            <Line
                              dataKey="p95"
                              stroke="var(--chart-1)"
                              strokeWidth={2}
                              dot={dot('var(--chart-1)')}
                              isAnimationActive={false}
                              connectNulls
                            />
                          </>
                        ) : (
                          <>
                            <Line
                              dataKey="p95"
                              stroke="var(--chart-2)"
                              strokeWidth={2}
                              dot={dot('var(--chart-2)')}
                              isAnimationActive={false}
                              connectNulls
                            />
                            <Line
                              dataKey="p50"
                              stroke="var(--chart-1)"
                              strokeWidth={2}
                              dot={dot('var(--chart-1)')}
                              isAnimationActive={false}
                              connectNulls
                            />
                          </>
                        )}
                      </LineChart>
                    </ChartContainer>
                  }
                  table={
                    <DataTable
                      caption="Latency by concurrency"
                      head={
                        comparing
                          ? ['Concurrency', 'This run p95 ms', 'Other run p95 ms']
                          : ['Concurrency', 'p50 ms', 'p95 ms']
                      }
                      rows={rows.map((r) =>
                        comparing
                          ? [r.c, fmtMs(r.p95, false), fmtMs(r.bp95, false)]
                          : [r.c, fmtMs(r.p50, false), fmtMs(r.p95, false)],
                      )}
                    />
                  }
                />

                <ChartPanel
                  title="Throughput vs concurrency"
                  description="× real time, by concurrent streams · higher is better"
                  legend={
                    comparing ? (
                      <>
                        <LegendKey color="var(--chart-1)" label={<span className="num">{seriesA}</span>} />
                        <LegendKey color="var(--chart-2)" label={<span className="num">{runLabel(compare)}</span>} />
                      </>
                    ) : undefined
                  }
                  chart={
                    <ChartContainer
                      config={latencyConfig}
                      className="aspect-auto h-64 w-full"
                      data-testid="bench-throughput"
                    >
                      <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 0, left: 0 }} accessibilityLayer>
                        <CartesianGrid {...gridProps} />
                        {xAxis}
                        <YAxis
                          {...axisProps}
                          axisLine={false}
                          width={48}
                          domain={[0, thrY.max]}
                          ticks={thrY.ticks}
                          tickFormatter={(v: number) => `${v}×`}
                        />
                        <ChartTooltip
                          cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
                          isAnimationActive={false}
                          content={({ active, payload }) => {
                            const r = payload?.[0]?.payload as Row | undefined
                            if (!active || !r) return null
                            const rowsTip = [
                              {
                                color: 'var(--chart-1)',
                                label: comparing ? 'this run' : 'throughput',
                                value: r.thr != null ? `${r.thr}×` : '–',
                              },
                            ]
                            if (comparing)
                              rowsTip.push({
                                color: 'var(--chart-2)',
                                label: 'other run',
                                value: r.bthr != null ? `${r.bthr}×` : '–',
                              })
                            return <TipBox title={`${r.c} concurrent`} rows={rowsTip} />
                          }}
                        />
                        {ceiling != null && !allPass && (
                          <ReferenceLine
                            x={ceiling}
                            stroke="var(--foreground)"
                            strokeOpacity={0.45}
                            strokeWidth={1}
                            label={{
                              value: `ceiling ${ceiling}`,
                              position: 'insideTopLeft',
                              fill: 'var(--foreground)',
                              fontSize: 11,
                            }}
                          />
                        )}
                        {comparing && (
                          <Line
                            dataKey="bthr"
                            stroke="var(--chart-2)"
                            strokeWidth={2}
                            dot={dot('var(--chart-2)')}
                            isAnimationActive={false}
                            connectNulls
                          />
                        )}
                        <Line
                          dataKey="thr"
                          stroke="var(--chart-1)"
                          strokeWidth={2}
                          dot={dot('var(--chart-1)')}
                          isAnimationActive={false}
                          connectNulls
                        />
                      </LineChart>
                    </ChartContainer>
                  }
                  table={
                    <DataTable
                      caption="Throughput by concurrency"
                      head={comparing ? ['Concurrency', 'This run ×', 'Other run ×'] : ['Concurrency', '× real time']}
                      rows={rows.map((r) => (comparing ? [r.c, r.thr ?? '–', r.bthr ?? '–'] : [r.c, r.thr ?? '–']))}
                    />
                  }
                />
              </div>

              <Panel title="Results" description="One row per concurrency level" bodyClassName="p-0 overflow-x-auto">
                <table className="w-full min-w-[640px] text-xs" data-testid="bench-table">
                  <thead>
                    <tr className="border-b border-border text-left text-muted-foreground">
                      {[
                        'Concurrency',
                        'OK',
                        'Errors',
                        `${BENCH_METRIC_LABEL[metric]} p50`,
                        `${BENCH_METRIC_LABEL[metric]} p95`,
                        'RTF p50',
                        'RTF p95',
                        'Throughput',
                        'Target',
                      ].map((h, i) => (
                        <th
                          key={h}
                          scope="col"
                          className={`px-4 py-2 font-medium whitespace-nowrap ${i > 0 && i < 8 ? 'text-right' : ''}`}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {current.points.map((pt) => {
                      const v = pt[metric]
                      const ok = v?.p95 != null && v.p95 <= p.targetMs
                      return (
                        <tr key={pt.concurrency} className="border-b border-border last:border-b-0">
                          <td className="px-4 py-2 num font-medium">{pt.concurrency}</td>
                          <td className="px-4 py-2 text-right num">{pt.ok}</td>
                          <td className="px-4 py-2 text-right num">{pt.errors}</td>
                          <td className="px-4 py-2 text-right num">{fmtMs(v?.p50)}</td>
                          <td className="px-4 py-2 text-right num">{fmtMs(v?.p95)}</td>
                          <td className="px-4 py-2 text-right num">{fmtRtf(pt.rtf?.p50)}</td>
                          <td className="px-4 py-2 text-right num">{fmtRtf(pt.rtf?.p95)}</td>
                          <td className="px-4 py-2 text-right num">{pt.throughput_x_realtime}×</td>
                          <td className="px-4 py-2">
                            <span className="inline-flex items-center gap-1.5">
                              <span
                                className={`size-1.5 rounded-full ${v?.p95 == null ? 'bg-chart-mute' : ok ? 'bg-good' : 'bg-critical'}`}
                                aria-hidden
                              />
                              {v?.p95 == null ? 'n/a' : ok ? 'met' : 'missed'}
                            </span>
                          </td>
                        </tr>
                      )
                    })}
                    {current.points.length === 0 && (
                      <tr>
                        <td colSpan={9} className="px-4 py-6 text-center text-muted-foreground">
                          Waiting for the first level to finish…
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </Panel>
            </>
          )}
        </div>
      </div>
    </Page>
  )
}
