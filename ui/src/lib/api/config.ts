/**
 * Where the API lives and how to reach it.
 *
 * The console is a static bundle served by the bolo server itself, possibly
 * under a sub-path, so by default the API base is the directory the page was
 * loaded from. `VITE_BOLO_SERVER` (build time) or the `bolo.server` local
 * storage key (run time) can point it at another origin.
 */

export interface Transport {
  fetch: typeof fetch
  WebSocket: typeof WebSocket
}

const KEY_STORAGE = 'bolo.apiKey'
const SERVER_STORAGE = 'bolo.server'

let transport: Transport = {
  fetch: (...args) => globalThis.fetch(...args),
  WebSocket: globalThis.WebSocket,
}

export function getTransport(): Transport {
  return transport
}

/** Replace the network transport (used by the mock layer and tests). */
export function setTransport(next: Partial<Transport>): void {
  transport = { ...transport, ...next }
}

function safeStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null
  } catch {
    return null
  }
}

export function pageBase(loc: Pick<Location, 'origin' | 'pathname'> = globalThis.location): string {
  // Hash routing keeps the pathname equal to the mount point (maybe with index.html).
  const dir = loc.pathname.replace(/[^/]*$/, '')
  return `${loc.origin}${dir}`
}

export function serverBase(): string {
  const stored = safeStorage()?.getItem(SERVER_STORAGE)
  const configured = stored || (import.meta.env.VITE_BOLO_SERVER as string | undefined)
  if (configured) return configured.endsWith('/') ? configured : `${configured}/`
  return pageBase()
}

export function setServerBase(url: string | null): void {
  const s = safeStorage()
  if (!s) return
  if (url) s.setItem(SERVER_STORAGE, url)
  else s.removeItem(SERVER_STORAGE)
}

/** Absolute http(s) URL for an API path such as `api/v1/info` or `/v1/models`. */
export function apiUrl(path: string, query?: Record<string, string | number | boolean | undefined | null>): string {
  const url = new URL(path.replace(/^\//, ''), serverBase())
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
    }
  }
  return url.toString()
}

/** Absolute ws(s) URL for a websocket path. The API key, if any, rides as `api_key`. */
export function wsUrl(path: string, query?: Record<string, string | number | boolean | undefined | null>): string {
  const key = getApiKey()
  const url = new URL(apiUrl(path, key ? { ...query, api_key: key } : query))
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export function getApiKey(): string | null {
  return safeStorage()?.getItem(KEY_STORAGE) || null
}

export function setApiKey(key: string | null): void {
  const s = safeStorage()
  if (!s) return
  if (key) s.setItem(KEY_STORAGE, key)
  else s.removeItem(KEY_STORAGE)
  globalThis.dispatchEvent?.(new Event('bolo:apikey'))
}

export function authHeaders(): Record<string, string> {
  const key = getApiKey()
  return key ? { Authorization: `Bearer ${key}` } : {}
}

export const isMock = import.meta.env.VITE_BOLO_MOCK === '1'

/** Built for static hosting with no server: everything runs in the browser (see src/demo/). */
export const isDemo = import.meta.env.VITE_BOLO_DEMO === '1'

/** How to get a real server; shown wherever the demo cannot do something. */
export const INSTALL_COMMAND =
  "pipx install 'bolo-speech[all] @ git+https://github.com/cb-Sigma/bolo#subdirectory=server'"
export const REPO_URL = 'https://github.com/cb-Sigma/bolo'
