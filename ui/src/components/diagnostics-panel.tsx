import { Bug, Check, Copy, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useStore } from '@/hooks/use-store'
import { sharedEngine, type EngineState } from '@/lib/browser-stt/engine'
import { diagnostics, environmentReport, type DiagnosticsState, type EnvironmentReport } from '@/lib/diagnostics'
import { fmtBytes } from '@/lib/format'
import { detectEnvironment } from '@/lib/mic-help'

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

/** The whole report as plain text, for pasting into an issue. */
export function diagnosticsText(env: EnvironmentReport | null, e: EngineState, d: DiagnosticsState): string {
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
  const refresh = useCallback(() => {
    void environmentReport().then(setEnv)
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh, d.errors.length, e.status])

  const copy = async () => {
    // No await before the clipboard call: Safari only allows it directly inside the tap.
    const text = diagnosticsText(env, e, d)
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
