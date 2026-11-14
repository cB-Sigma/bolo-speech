import { TriangleAlert } from 'lucide-react'
import { useMemo } from 'react'
import { LegendKey } from '@/components/charts/chart-kit'
import { Empty } from '@/components/panel'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import type { RequestTrace } from '@/lib/api/types'
import { fmtClock, fmtMs, fmtRtf, fmtSeconds } from '@/lib/format'
import { niceStep, percentile } from '@/lib/latency/stats'
import { BROWSER_STT_PHASES, phaseColor, phaseLabel, phasesFor, traceExtras } from '@/lib/phases'
import { romanisedSummary } from '@/lib/romanised'
import { cn } from '@/lib/utils'
import { ListTree } from 'lucide-react'
import { OverrideText } from '@/components/wordlists/override-text'

function Bars({ trace, scaleMs, height = 'h-3' }: { trace: RequestTrace; scaleMs: number; height?: string }) {
  const pct = (ms: number) => `${Math.min(100, (ms / scaleMs) * 100)}%`
  const marker = trace.kind === 'tts' ? trace.ttfa_ms : trace.first_partial_ms
  return (
    <div className="relative h-5 w-full">
      {trace.phases.map((p, i) => (
        <div
          key={i}
          className={cn('absolute top-1 rounded-[2px]', height)}
          style={{
            left: pct(p.start_ms),
            display: p.start_ms >= scaleMs ? 'none' : undefined,
            // 2px surface gap between neighbouring segments
            width: `max(2px, calc(${pct(Math.min(p.dur_ms, scaleMs - p.start_ms))} - 2px))`,
            background: phaseColor(trace.kind, p.name),
          }}
        />
      ))}
      {marker != null && marker <= scaleMs && (
        <div className="absolute top-0 h-5 w-px bg-foreground" style={{ left: pct(marker) }} />
      )}
      {trace.total_ms > scaleMs && (
        <div
          className="absolute top-0 right-0 flex h-5 items-center bg-linear-to-r from-transparent to-card pl-4"
          aria-hidden
        >
          <span className="num text-[10px] leading-none text-muted-foreground">››</span>
        </div>
      )}
    </div>
  )
}

/**
 * Recent requests as rows on one shared duration axis. Each row's phases are
 * drawn where they happened, so queueing shows up as a bar that starts late.
 */
export function Waterfall({
  traces,
  onSelect,
  selectedId,
  kind,
}: {
  traces: RequestTrace[]
  onSelect: (t: RequestTrace) => void
  selectedId: string | null
  kind: 'stt' | 'tts' | 'mixed'
}) {
  // A few very long requests (a paragraph of TTS) would flatten every other
  // row, so the axis covers the typical request and longer bars are cut off
  // with a break mark. The Total column always shows the real figure.
  const scaleMs = useMemo(() => {
    const totals = traces.map((t) => t.total_ms)
    const typical = Math.max(100, percentile(totals, 80) * 1.3)
    const max = Math.min(Math.max(100, ...totals), typical)
    const step = niceStep(max, 4)
    return Math.ceil(max / step) * step
  }, [traces])
  const ticks = useMemo(() => {
    const step = niceStep(scaleMs, 4)
    return Array.from({ length: Math.floor(scaleMs / step + 1e-9) + 1 }, (_, i) => i * step)
  }, [scaleMs])

  if (traces.length === 0) {
    return (
      <Empty icon={ListTree} title="No requests in this window">
        Requests appear here as they complete. Send some audio from the playground, or widen the time range.
      </Empty>
    )
  }
  const legendKinds = kind === 'mixed' ? (['stt', 'tts'] as const) : ([kind] as const)
  // The in-browser recogniser has its own, shorter pipeline.
  const browserOnly = traces.every((t) => t.transport === 'browser')
  const cols = 'grid-cols-[4.5rem_minmax(0,1fr)_3.5rem] sm:grid-cols-[4.5rem_10.5rem_minmax(0,1fr)_4rem]'
  return (
    <div className="min-w-0">
      <div className="mb-3 grid gap-1 text-xs text-muted-foreground">
        {legendKinds.map((k) => (
          <div key={k} className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {kind === 'mixed' && <span className="w-7 font-medium text-foreground uppercase">{k}</span>}
            {(browserOnly ? BROWSER_STT_PHASES : phasesFor(k)).map((name) => (
              <LegendKey key={name} shape="rect" color={phaseColor(k, name)} label={phaseLabel(name)} />
            ))}
            <span className="inline-flex items-center gap-1.5">
              <span className="h-3 w-px bg-foreground" aria-hidden />
              {k === 'tts' ? 'first audio' : 'first partial'}
            </span>
          </div>
        ))}
      </div>
      <div className={cn('grid items-end gap-x-3 border-b border-border pb-1 text-[11px] text-muted-foreground', cols)}>
        <span>Started</span>
        <span className="max-sm:hidden">Model</span>
        <div className="relative h-4" aria-label="Milliseconds since the request started">
          {ticks.map((t, i) => (
            <span
              key={t}
              className={cn(
                'absolute bottom-0 num',
                i === 0 ? '' : i === ticks.length - 1 && t === scaleMs ? '-translate-x-full' : '-translate-x-1/2',
              )}
              style={{ left: `${(t / scaleMs) * 100}%` }}
            >
              {t}
            </span>
          ))}
        </div>
        <span className="text-right">Total ms</span>
      </div>
      <ul className="relative" data-testid="waterfall">
        {traces.map((t) => (
          <li key={t.id}>
            <button
              type="button"
              onClick={() => onSelect(t)}
              className={cn(
                'grid w-full items-center gap-x-3 border-b border-border/60 py-1 text-left text-xs transition-colors duration-150 hover:bg-muted/50 focus-visible:bg-muted/50',
                cols,
                selectedId === t.id && 'bg-muted/60',
              )}
            >
              <span className="num text-muted-foreground">
                <span className="sr-only">{`Request ${t.id}, started `}</span>
                {fmtClock(t.started_at)}
              </span>
              <span className="flex min-w-0 items-center gap-1.5 max-sm:hidden">
                <span className="w-6 shrink-0 text-[10px] font-medium text-muted-foreground uppercase">{` ${t.kind} `}</span>
                <span className="truncate num">{t.model}</span>
              </span>
              <div className="relative min-w-0">
                <div className="pointer-events-none absolute inset-0">
                  {ticks.slice(1).map((tick) => (
                    <div
                      key={tick}
                      className="absolute inset-y-0 w-px bg-chart-grid"
                      style={{ left: `${(tick / scaleMs) * 100}%` }}
                    />
                  ))}
                </div>
                <Bars trace={t} scaleMs={scaleMs} />
              </div>
              <span className="flex items-center justify-end gap-1 text-right num">
                {t.status !== 'ok' && (
                  <>
                    <TriangleAlert className="size-3 text-critical-text" aria-hidden />
                    <span className="sr-only">failed, </span>
                  </>
                )}
                {Math.round(t.total_ms)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

const KIND_LABEL: Record<string, string> = {
  stt: 'Speech to text',
  tts: 'Text to speech',
  vad: 'Voice activity',
  diarization: 'Diarization',
  agent: 'Voice agent turn',
}

export function RequestDrawer({ trace, onClose }: { trace: RequestTrace | null; onClose: () => void }) {
  return (
    <Sheet open={trace != null} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        className="w-full gap-0 overflow-y-auto sm:max-w-md"
        // Focus the panel itself, not its first tooltip trigger.
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          ;(e.currentTarget as HTMLElement | null)?.focus()
        }}
        data-testid="request-drawer"
      >
        {trace && (
          <>
            <SheetHeader className="border-b border-border">
              <SheetTitle className="num text-base">{trace.id}</SheetTitle>
              <SheetDescription>
                {KIND_LABEL[trace.kind] ?? trace.kind} · {trace.transport} · {fmtClock(trace.started_at)}
              </SheetDescription>
            </SheetHeader>
            <div className="grid gap-5 p-4">
              {trace.status !== 'ok' && (
                <p
                  role="alert"
                  className="flex items-start gap-2 rounded-lg border border-critical/30 bg-critical/10 p-3 text-[13px]"
                >
                  <TriangleAlert className="mt-0.5 size-4 shrink-0 text-critical-text" aria-hidden />
                  <span>
                    <span className="font-medium">Failed.</span> {trace.error ?? 'No error message was recorded.'}
                  </span>
                </p>
              )}
              <section>
                <h3 className="mb-2 eyebrow">Phases</h3>
                <Bars trace={trace} scaleMs={Math.max(1, trace.total_ms)} height="h-3" />
                <table className="mt-2 w-full text-xs">
                  <thead>
                    <tr className="text-left text-muted-foreground">
                      <th scope="col" className="py-1 font-medium">
                        Phase
                      </th>
                      <th scope="col" className="py-1 text-right font-medium">
                        Starts
                      </th>
                      <th scope="col" className="py-1 text-right font-medium">
                        Takes
                      </th>
                      <th scope="col" className="w-12 py-1 text-right font-medium">
                        Share
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {trace.phases.map((p, i) => (
                      <tr key={i} className="border-t border-border">
                        <td className="py-1.5">
                          <span className="flex items-center gap-2">
                            <span
                              className="size-2.5 rounded-[2px]"
                              style={{ background: phaseColor(trace.kind, p.name) }}
                              aria-hidden
                            />
                            {phaseLabel(p.name)}
                          </span>
                        </td>
                        <td className="py-1.5 text-right num text-muted-foreground">{fmtMs(p.start_ms)}</td>
                        <td className="py-1.5 text-right num">{fmtMs(p.dur_ms)}</td>
                        <td className="py-1.5 text-right num text-muted-foreground">
                          {trace.total_ms > 0 ? `${Math.round((p.dur_ms / trace.total_ms) * 100)}%` : '–'}
                        </td>
                      </tr>
                    ))}
                    <tr className="border-t border-border font-medium">
                      <td className="py-1.5">Total</td>
                      <td />
                      <td className="py-1.5 text-right num">{fmtMs(trace.total_ms)}</td>
                      <td />
                    </tr>
                  </tbody>
                </table>
              </section>
              <section>
                <h3 className="mb-2 eyebrow">Timing</h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  {trace.kind === 'tts' ? (
                    <>
                      <dt className="text-muted-foreground">Time to first audio</dt>
                      <dd className="text-right num">{fmtMs(trace.ttfa_ms)}</dd>
                    </>
                  ) : (
                    <>
                      <dt className="text-muted-foreground">First partial</dt>
                      <dd className="text-right num">{fmtMs(trace.first_partial_ms)}</dd>
                      <dt className="text-muted-foreground">Final</dt>
                      <dd className="text-right num">{fmtMs(trace.final_ms)}</dd>
                    </>
                  )}
                  <dt className="text-muted-foreground">Real-time factor</dt>
                  <dd className="text-right num">{fmtRtf(trace.rtf)}</dd>
                  <dt className="text-muted-foreground">Audio</dt>
                  <dd className="text-right num">{fmtSeconds(trace.audio_seconds, 2)}</dd>
                </dl>
              </section>
              <section>
                <h3 className="mb-2 eyebrow">Request</h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  <dt className="text-muted-foreground">Model</dt>
                  <dd className="truncate text-right num">{trace.model}</dd>
                  <dt className="text-muted-foreground">Transport</dt>
                  <dd className="text-right">{trace.transport}</dd>
                  <dt className="text-muted-foreground">Language</dt>
                  <dd className="text-right">{trace.language ?? '–'}</dd>
                  {trace.chars != null && (
                    <>
                      <dt className="text-muted-foreground">Characters</dt>
                      <dd className="text-right num">{trace.chars}</dd>
                    </>
                  )}
                  <dt className="text-muted-foreground">Batch size</dt>
                  <dd className="text-right num">{trace.batch_size ?? '–'}</dd>
                  <dt className="text-muted-foreground">Started</dt>
                  <dd className="text-right num">
                    {new Date(trace.started_at).toISOString().replace('T', ' ').replace('Z', ' UTC')}
                  </dd>
                </dl>
              </section>
              {traceExtras(trace).length > 0 && (
                <section data-testid="trace-extras">
                  <h3 className="mb-2 eyebrow">Speakers, separation and pipeline</h3>
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                    {traceExtras(trace).map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-muted-foreground">{k}</dt>
                        <dd className="truncate text-right num">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}
              {trace.text_preview && (
                <section>
                  <h3 className="mb-2 eyebrow">Text</h3>
                  <p className="rounded-lg border border-border bg-muted/40 p-3 text-[13px] leading-5">
                    {trace.text_preview}
                  </p>
                </section>
              )}
              {(trace.normalized_text || (trace.pronunciations_applied?.length ?? 0) > 0 || trace.romanised) && (
                <section data-testid="trace-text-prep">
                  <h3 className="mb-2 eyebrow">
                    What the voice received{trace.normalizer ? ` (normaliser ${trace.normalizer})` : ''}
                  </h3>
                  {trace.normalized_text && (
                    <OverrideText text={trace.normalized_text} className="text-[13px] leading-6" />
                  )}
                  {trace.romanised && (
                    <p
                      className={cn(
                        'mt-2 text-xs',
                        trace.romanised.applied ? 'text-muted-foreground' : 'text-warning-text',
                      )}
                      data-testid="trace-romanised"
                    >
                      {romanisedSummary(trace.romanised)}
                    </p>
                  )}
                  {(trace.pronunciations_applied?.length ?? 0) > 0 && (
                    <ul className="mt-2 grid gap-1 text-xs" aria-label="Pronunciations applied">
                      {trace.pronunciations_applied?.map((a, i) => (
                        <li key={i}>
                          <span className="text-foreground">{a.text}</span> →{' '}
                          <span className="num">{a.spoken ?? (a.ipa ? `/${a.ipa}/` : '?')}</span>
                          {a.word && a.word !== a.text && (
                            <span className="text-muted-foreground"> (entry {a.word})</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
