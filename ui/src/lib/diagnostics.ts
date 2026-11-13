import type { FetchRecord } from './browser-stt/models'
import { Store } from './store'

/**
 * A running record of what the page did while loading models and opening the
 * microphone. Always collected (it is tiny); shown by the diagnostics panel
 * when the page is opened with `?debug=1`.
 */
export interface DiagnosticEvent {
  /** Milliseconds since the page started. */
  at: number
  source: string
  message: string
}

export interface DiagnosticsState {
  fetches: FetchRecord[]
  errors: DiagnosticEvent[]
  events: DiagnosticEvent[]
}

const LIMIT = 80

export const diagnostics = new Store<DiagnosticsState>({ fetches: [], errors: [], events: [] })

const at = () => Math.round(typeof performance === 'undefined' ? 0 : performance.now())
const push = <T>(list: T[], item: T): T[] => [...list.slice(-(LIMIT - 1)), item]

export const diag = {
  fetch(record: FetchRecord): void {
    diagnostics.patch({ fetches: push(diagnostics.get().fetches, record) })
  },
  error(source: string, message: string): void {
    diagnostics.patch({ errors: push(diagnostics.get().errors, { at: at(), source, message }) })
  },
  event(source: string, message: string): void {
    diagnostics.patch({ events: push(diagnostics.get().events, { at: at(), source, message }) })
  },
}

const DEBUG_KEY = 'bolo.debug'

/** True when the page was opened with `?debug=1` (before or after the `#`); sticks for the tab. */
export function isDebug(loc: Pick<Location, 'search' | 'hash'> = globalThis.location): boolean {
  const has = (q: string) => /(^|[?&])debug=1(&|$)/.test(q)
  const off = (q: string) => /(^|[?&])debug=0(&|$)/.test(q)
  const hashQuery = loc.hash.includes('?') ? loc.hash.slice(loc.hash.indexOf('?')) : ''
  try {
    if (off(loc.search) || off(hashQuery)) {
      sessionStorage.removeItem(DEBUG_KEY)
      return false
    }
    if (has(loc.search) || has(hashQuery)) {
      sessionStorage.setItem(DEBUG_KEY, '1')
      return true
    }
    return sessionStorage.getItem(DEBUG_KEY) === '1'
  } catch {
    return has(loc.search) || has(hashQuery)
  }
}

let installed = false

/** Record uncaught errors and rejected promises. */
export function installErrorCapture(): void {
  if (installed || typeof window === 'undefined') return
  installed = true
  window.addEventListener('error', (e) => diag.error('window', e.message || String(e.error ?? 'error')))
  window.addEventListener('unhandledrejection', (e) => {
    const r: unknown = e.reason
    diag.error('promise', r instanceof Error ? `${r.name}: ${r.message}` : String(r))
  })
}

export interface EnvironmentReport {
  userAgent: string
  url: string
  secureContext: boolean
  crossOriginIsolated: boolean
  sharedArrayBuffer: boolean
  hardwareConcurrency: number
  coep: string
  serviceWorker: string
  microphonePermission: string
  connection: string
  online: boolean
}

/** Snapshot of the things that decide whether the microphone and the in-browser model can work. */
export async function environmentReport(): Promise<EnvironmentReport> {
  const nav = navigator as Navigator & {
    connection?: { effectiveType?: string; saveData?: boolean; downlink?: number; type?: string }
  }
  let sw = 'unsupported'
  let coep = globalThis.crossOriginIsolated ? 'set by the server (response headers)' : 'none'
  if ('serviceWorker' in nav) {
    try {
      const reg = await nav.serviceWorker.getRegistration()
      const worker = reg?.active ?? reg?.waiting ?? reg?.installing
      sw = !reg
        ? 'not registered'
        : `${worker?.state ?? 'no worker'}${nav.serviceWorker.controller ? ', controlling this page' : ', not controlling this page'}`
      const mode = worker ? new URL(worker.scriptURL).searchParams.get('coep') : null
      if (mode) coep = `${mode} (added by the service worker)`
    } catch (err) {
      sw = `error: ${err instanceof Error ? err.message : String(err)}`
    }
  }
  let permission = 'unknown (the browser does not report it)'
  try {
    permission = (await nav.permissions.query({ name: 'microphone' as PermissionName })).state
  } catch {
    // Safari before 16 and some webviews cannot be asked.
  }
  const c = nav.connection
  return {
    userAgent: nav.userAgent,
    url: globalThis.location.href,
    secureContext: Boolean(globalThis.isSecureContext),
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    hardwareConcurrency: nav.hardwareConcurrency ?? 1,
    coep,
    serviceWorker: sw,
    microphonePermission: permission,
    connection: c
      ? `${c.effectiveType ?? c.type ?? 'unknown'}${c.downlink != null ? `, ~${c.downlink} Mbit/s` : ''}${c.saveData ? ', data saver on' : ''}`
      : 'unknown (the browser does not report it)',
    online: nav.onLine,
  }
}

/** True when the browser says the user wants to save data, or the link is slow enough that 30 MB matters. */
export function prefersLessData(): boolean {
  const c = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection
  return Boolean(c?.saveData) || /(^|-)2g$/.test(c?.effectiveType ?? '')
}
