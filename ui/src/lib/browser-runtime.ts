import type { FetchRecord } from '@/lib/browser-stt/models'
import { diag } from '@/lib/diagnostics'

/**
 * Download helpers shared by the in-browser recogniser and synthesiser workers:
 * fetch a file with byte-level progress, keep it in Cache Storage, and report
 * the request for the diagnostics panel.
 */

export interface DownloadHooks {
  /** Bytes so far, and the total when the server sent Content-Length (else 0). */
  progress?: (loaded: number, total: number) => void
  record?: (record: FetchRecord) => void
}

async function readWithProgress(
  res: Response,
  hooks: DownloadHooks,
): Promise<{ chunks: Uint8Array[]; loaded: number }> {
  const total = Number(res.headers.get('content-length')) || 0
  hooks.progress?.(0, total)
  const reader = res.body?.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  if (!reader) {
    const buf = new Uint8Array(await res.arrayBuffer())
    hooks.progress?.(buf.byteLength, buf.byteLength)
    return { chunks: [buf], loaded: buf.byteLength }
  }
  let last = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    const now = performance.now()
    if (now - last > 100) {
      last = now
      hooks.progress?.(loaded, total)
    }
  }
  hooks.progress?.(loaded, total || loaded)
  return { chunks, loaded }
}

/**
 * The file at `url` from Cache Storage (`cacheName`, under `key`, default the
 * URL), downloading it with progress the first time. Without Cache Storage
 * (some private windows) it is simply downloaded.
 */
export async function cachedDownload(
  url: string,
  cacheName: string,
  hooks: DownloadHooks = {},
  opts: { key?: string; contentType?: string } = {},
): Promise<{ data: ArrayBuffer; fromCache: boolean }> {
  const key = opts.key ?? url
  let cache: Cache | null
  try {
    cache = typeof caches === 'undefined' ? null : await caches.open(cacheName)
    const hit = await cache?.match(key)
    if (hit) {
      const data = await hit.arrayBuffer()
      hooks.progress?.(data.byteLength, data.byteLength)
      return { data, fromCache: true }
    }
  } catch {
    cache = null
  }
  const started = performance.now()
  const record: FetchRecord = { url, status: 0, bytes: null, ms: 0 }
  try {
    const res = await fetch(url)
    record.status = res.status
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
    const { chunks, loaded } = await readWithProgress(res, hooks)
    record.bytes = loaded
    // One contiguous copy; the pieces are dropped right away (a voice is tens of MB, phones are short of memory).
    const bytes = new Uint8Array(loaded)
    let o = 0
    for (const c of chunks.splice(0)) {
      bytes.set(c, o)
      o += c.byteLength
    }
    if (cache) {
      try {
        await cache.put(
          key,
          new Response(bytes, {
            status: 200,
            headers: {
              'content-type': opts.contentType ?? 'application/octet-stream',
              'content-length': String(loaded),
            },
          }),
        )
      } catch {
        // Quota or private mode: carry on without keeping a copy.
      }
    }
    return { data: bytes.buffer, fromCache: false }
  } catch (err) {
    record.error = err instanceof Error ? err.message : String(err)
    throw err
  } finally {
    record.ms = Math.round(performance.now() - started)
    hooks.record?.(record)
  }
}

/** jsDelivr directory of the onnxruntime-web build with this version. */
export function ortCdnBase(version: string): string {
  return `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`
}

/** Cache bucket that @huggingface/transformers keeps the runtime binary in; shared so it is fetched once. */
export const RUNTIME_CACHE = 'transformers-cache'

/**
 * ONNX Runtime Web (1.19 and later) ships only builds that create a *shared*
 * WebAssembly.Memory with a 4 GiB maximum, even on one thread. Engines reserve
 * address space for the whole maximum of a shared memory up front. WebKit on
 * iOS has little of it: one such memory fits, a second one in the same page
 * fails with "RangeError: Out of memory". The synthesiser therefore loads the
 * runtime's JavaScript with a smaller maximum (the WebAssembly module accepts
 * any maximum up to its own), and on WebKit only one in-browser model runtime
 * is kept alive at a time (see claimRuntime).
 */
export const TTS_MEMORY_PAGES = 16_384 // 1 GiB; a 13-second sentence of a 114 MB VITS voice needs about 0.3 GiB

const GLUE_MEMORY = /new WebAssembly\.Memory\(\{initial:(\d+),maximum:65536,shared:!0\}\)/

/**
 * The runtime's loader with the memory maximum lowered to `pages`, and the
 * memory left on `globalThis.__boloOrtMemory` so its size can be reported;
 * null when the loader does not look as expected.
 */
export function lowerMemoryMaximum(source: string, pages: number): string | null {
  if (!GLUE_MEMORY.test(source)) return null
  return source.replace(
    GLUE_MEMORY,
    (_m, initial: string) =>
      `(globalThis.__boloOrtMemory=new WebAssembly.Memory({initial:${initial},maximum:${pages},shared:!0}))`,
  )
}

/** Bytes currently in the runtime's WebAssembly memory (only known when the loader was lowered). */
export function runtimeMemoryBytes(): number | null {
  const m = (globalThis as { __boloOrtMemory?: WebAssembly.Memory }).__boloOrtMemory
  return m ? m.buffer.byteLength : null
}

/** True for errors that mean the browser could not give WebAssembly the memory it asked for. */
export function isOutOfMemory(message: string | null | undefined): boolean {
  return /out of memory|could not allocate memory|Array buffer allocation failed|memory access out of bounds/i.test(
    message ?? '',
  )
}

/** WebKit (every browser on iPhone and iPad, and Safari): tight on address space for WebAssembly memories. */
export function isWebKit(
  nav: Pick<Navigator, 'userAgent' | 'maxTouchPoints'> | undefined = globalThis.navigator,
): boolean {
  if (!nav) return false
  const ua = nav.userAgent
  const ios = /iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && nav.maxTouchPoints > 1)
  const safari = /AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg\/|Firefox|OPR\//.test(ua)
  return ios || safari
}

let exclusive: boolean | null = null
const holders = new Map<string, () => void>()

/** Whether only one in-browser model runtime may be alive in the page at a time. */
export function oneRuntimeAtATime(): boolean {
  exclusive ??= isWebKit()
  return exclusive
}

/** After an out-of-memory error: from now on keep one runtime at a time, whatever the browser. */
export function requireOneRuntime(): void {
  exclusive = true
}

/**
 * Register a runtime that is about to load. When only one may be alive, every
 * other holder is released first (its worker terminated, its memory freed);
 * it loads again by itself the next time it is needed.
 */
export function claimRuntime(name: string, release: () => void): void {
  if (oneRuntimeAtATime()) {
    for (const [other, free] of [...holders]) {
      if (other === name) continue
      holders.delete(other)
      free()
    }
  }
  holders.set(name, release)
}

export function releaseRuntime(name: string): void {
  holders.delete(name)
}

/** Release every runtime but `except` (used before retrying after an out-of-memory error). */
export function releaseOtherRuntimes(except: string): string[] {
  const freed: string[] = []
  for (const [other, free] of [...holders]) {
    if (other === except) continue
    holders.delete(other)
    free()
    freed.push(other)
  }
  return freed
}

/**
 * Log how much memory the page uses, for the diagnostics panel: the precise
 * figure where the browser offers it (Chromium, cross-origin isolated), the
 * JavaScript heap otherwise, and the device's memory class when known.
 */
export async function reportPageMemory(when: string): Promise<void> {
  const mb = (b: number) => `${(b / 2 ** 20).toFixed(0)} MiB`
  const parts: string[] = []
  const perf = globalThis.performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>
    memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number }
  }
  try {
    if (perf.measureUserAgentSpecificMemory && globalThis.crossOriginIsolated) {
      parts.push(`page ${mb((await perf.measureUserAgentSpecificMemory()).bytes)}`)
    } else if (perf.memory) {
      parts.push(`JS heap ${mb(perf.memory.usedJSHeapSize)} of ${mb(perf.memory.jsHeapSizeLimit)}`)
    }
  } catch {
    // Not available here.
  }
  const device = (globalThis.navigator as (Navigator & { deviceMemory?: number }) | undefined)?.deviceMemory
  if (device) parts.push(`device about ${device} GB`)
  parts.push(oneRuntimeAtATime() ? 'one model runtime at a time' : 'runtimes may coexist')
  diag.event('memory', `${when}: ${parts.join(' · ')}`)
}
