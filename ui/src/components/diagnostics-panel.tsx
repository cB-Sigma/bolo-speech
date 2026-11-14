import { Bug, Check, Copy, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useStore } from '@/hooks/use-store'
import { sharedEngine, type EngineState } from '@/lib/browser-stt/engine'
import { diagnostics, environmentReport, type DiagnosticsState, type EnvironmentReport } from '@/lib/diagnostics'
import { fmtBytes } from '@/lib/format'
import { detectEnvironment } from '@/lib/mic-help'
import { useInfo, useLive, useWordLists } from '@/hooks/use-api'
import type { RequestTrace, ServerInfo } from '@/lib/api/types'
import { traceExtras } from '@/lib/phases'

function engineLines(e: EngineState): [string, string][] {
  return [
    ['Model', `${e.model.id} (${e.model.repo}, ${e.model.dtype})`],
    [
      'Model status',
      `${e.status}${e.stage ? `, ${e.stage}` : ''}${e.stalled ? `, no progress for ${e.stalled.seconds} s` : ''}`,
    ],
    [
      'Runs on',
      e.device
        ? `${e.device}, ${e.threads} thread${e.threads === 1 ? '' : 's'}${e.safeMode ? ' (single-threaded retry)' : ''}`
        : 'not started',
    ],
    ['Model in cache', e.cached == null ? 'unknown' : e.cached ? 'yes' : 'no'],
    ['Load time', e.loadMs != null ? `${Math.round(e.loadMs)} ms` : 'n/a'],
    ['Model error', e.error ?? 'none'],
  ]
}

function envLines(env: EnvironmentReport): [string, string][] {
  const d = detectEnvironment(env.userAgent)
  return [
    ['User agent', env.userAgent],
    ['Detected', `${d.platform}, ${d.browser}${d.inApp ? `, in-app browser (${d.inApp})` : ''}`],
    ['Page', env.url],
    ['Secure context', env.secureContext ? 'yes' : 'no'],
    ['crossOriginIsolated', env.crossOriginIsolated ? 'yes' : 'no'],
    ['SharedArrayBuffer', env.sharedArrayBuffer ? 'available' : 'unavailable'],
    ['Logical cores', String(env.hardwareConcurrency)],
    ['COEP mode', env.coep],
    ['Service worker', env.serviceWorker],
    ['Microphone permission', env.microphonePermission],
    ['Connection', `${env.connection}${env.online ? '' : ' (offline)'}`],
  ]
}

/** What the server says it can do: realtime pipelines, the diarizer, separation. */
export function serverLines(info: ServerInfo | undefined): [string, string][] {
  if (!info) return [['Server', 'not reached']]
  const c = info.capabilities
  if (!c) return [['Server', `${info.name} ${info.version} (reports no capabilities)`]]
  const pipes = Object.entries(c.realtime?.pipelines ?? {})
    .map(([id, p]) => {
      const ok = p?.available ?? p?.configured ?? Boolean(p?.url)
      return `${id}${ok ? '' : ' (not set up)'}`
    })
    .join(', ')
  return [
    ['Server', `${info.name} ${info.version}`],
    ['Realtime pipeline', `${c.realtime?.pipeline ?? 'not reported'}${pipes ? `; offers ${pipes}` : ''}`],
    [
      'Default diarizer',
      `${c.diarization?.default_model ?? 'none'}${c.diarization?.installed ? ', installed' : ', not installed'}${c.diarization?.speaker_conditioned_models?.length ? `; speaker-conditioned: ${c.diarization.speaker_conditioned_models.join(', ')}` : ''}`,
    ],
    [
      'Default separation',
      c.separation
        ? `${c.separation.default_model ?? 'none'}${c.separation.available ? '' : ' (unavailable)'}${c.separation.installed ? ', installed' : ''}`
        : 'not reported',
    ],
  ]
}

/** The newest server trace with diarization, separation or pipeline fields. */
export function lastTraceLines(requests: RequestTrace[]): [string, string][] {
  const t = requests.find((r) => traceExtras(r).length > 0)
  if (!t) return []
  return [['Request', `${t.id} (${t.kind}, ${t.model})`], ...traceExtras(t)]
}

/** The whole report as plain text, for pasting into an issue. */
export function diagnosticsText(
  env: EnvironmentReport | null,
  e: EngineState,
  d: DiagnosticsState,
  server: [string, string][] = [],
): string {
  const rows = (pairs: [string, string][]) => pairs.map(([k, v]) => `${k}: ${v}`).join('\n')
  const fetches = d.fetches.length
    ? d.fetches
        .map(
          (f) =>
            `${f.status || 'ERR'}  ${f.bytes != null ? fmtBytes(f.bytes) : '?'}  ${f.ms} ms  ${f.url}${f.error ? `  (${f.error})` : ''}`,
        )
        .join('\n')
    : 'none yet (files already in the browser cache are not fetched)'
  const list = (items: DiagnosticsState['errors']) =>
    items.length ? items.map((x) => `+${(x.at / 1000).toFixed(1)}s  ${x.source}: ${x.message}`).join('\n') : 'none'
  return [
    'bolo diagnostics',
    env ? rows(envLines(env)) : 'environment: collecting',
    '',
    rows(engineLines(e)),
    '',
    ...(server.length ? [rows(server), ''] : []),
    'Fetches (status, size, time to response, URL):',
    fetches,
    '',
    'Errors:',
    list(d.errors),
    '',
    'Events:',
    list(d.events),
  ].join('\n')
}

function Rows({ pairs }: { pairs: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[minmax(7rem,auto)_minmax(0,1fr)] gap-x-3 gap-y-0.5">
      {pairs.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="min-w-0 num break-words">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Shown at the bottom of every screen when the page is opened with `?debug=1`. */
export function DiagnosticsPanel() {
  const d = useStore(diagnostics)
  const e = useStore(sharedEngine().store)
  const [env, setEnv] = useState<EnvironmentReport | null>(null)
  const [copied, setCopied] = useState(false)
  const info = useInfo().data
  const requests = useLive().requests
  const lists = useWordLists()
  const server: [string, string][] = [
    ...serverLines(info),
    [
      'Word lists',
      lists.offered
        ? `${lists.data?.wordlists.length ?? 0} list(s)${lists.data?.available ? '' : ', uccharan missing'}`
        : 'not offered',
    ],
    ...lastTraceLines(requests),
  ]
  const refresh = useCallback(() => {
    void environmentReport().then(setEnv)
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh, d.errors.length, e.status])

  const copy = async () => {
    // No await before the clipboard call: Safari only allows it directly inside the tap.
    const text = diagnosticsText(env, e, d, server)
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Could not copy. Select the text below instead.')
    }
  }

  return (
    <section
      aria-labelledby="diagnostics-title"
      className="mx-auto mb-6 w-full max-w-[1400px] px-4 sm:px-6"
      data-testid="diagnostics"
    >
      <div className="rounded-xl border border-border bg-card text-xs leading-5">
        <header className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
          <Bug className="size-4 text-muted-foreground" aria-hidden />
          <h2 id="diagnostics-title" className="flex-1 text-[13px] font-semibold">
            Diagnostics
          </h2>
          <Button size="sm" variant="ghost" onClick={refresh}>
            <RefreshCw /> Refresh
          </Button>
          <Button size="sm" variant="outline" onClick={() => void copy()} data-testid="diagnostics-copy">
            {copied ? <Check /> : <Copy />} {copied ? 'Copied' : 'Copy'}
          </Button>
        </header>
        <div className="grid gap-4 p-4 lg:grid-cols-2">
          <div className="min-w-0" data-testid="diagnostics-env">
            <h3 className="mb-1 eyebrow">Browser</h3>
            {env ? <Rows pairs={envLines(env)} /> : <p className="text-muted-foreground">Collecting…</p>}
          </div>
          <div className="min-w-0" data-testid="diagnostics-engine">
            <h3 className="mb-1 eyebrow">In-browser model</h3>
            <Rows pairs={engineLines(e)} />
          </div>
          <div className="min-w-0 lg:col-span-2" data-testid="diagnostics-server">
            <h3 className="mb-1 eyebrow">Server features and last trace</h3>
            <Rows pairs={server} />
          </div>
          <div className="min-w-0 lg:col-span-2" data-testid="diagnostics-fetches">
            <h3 className="mb-1 eyebrow">Model and runtime requests</h3>
            {d.fetches.length === 0 ? (
              <p className="text-muted-foreground">
                None yet. Files already in the browser cache are not requested again.
              </p>
            ) : (
              <ul className="space-y-0.5">
                {d.fetches.map((f, i) => (
                  <li key={i} className="flex flex-wrap gap-x-3 num">
                    <span className={f.status >= 200 && f.status < 400 ? 'text-good-text' : 'text-critical-text'}>
                      {f.status || 'ERR'}
                    </span>
                    <span>{f.bytes != null ? fmtBytes(f.bytes) : 'size unknown'}</span>
                    <span>{f.ms} ms</span>
                    <span className="min-w-0 break-all text-muted-foreground">
                      {f.url}
                      {f.error ? ` (${f.error})` : ''}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="min-w-0" data-testid="diagnostics-errors">
            <h3 className="mb-1 eyebrow">Errors</h3>
            {d.errors.length === 0 ? (
              <p className="text-muted-foreground">None.</p>
            ) : (
              <ul className="space-y-0.5">
                {d.errors.map((x, i) => (
                  <li key={i} className="break-words">
                    <span className="num text-muted-foreground">+{(x.at / 1000).toFixed(1)}s</span> {x.source}:{' '}
                    {x.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="min-w-0" data-testid="diagnostics-events">
            <h3 className="mb-1 eyebrow">Events</h3>
            {d.events.length === 0 ? (
              <p className="text-muted-foreground">None yet.</p>
            ) : (
              <ul className="space-y-0.5">
                {d.events.slice(-20).map((x, i) => (
                  <li key={i} className="break-words">
                    <span className="num text-muted-foreground">+{(x.at / 1000).toFixed(1)}s</span> {x.source}:{' '}
                    {x.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
