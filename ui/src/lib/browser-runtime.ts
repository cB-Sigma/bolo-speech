/// <reference lib="webworker" />
import type { FetchRecord } from '@/lib/browser-stt/models'

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
    const blob = new Blob(chunks as BlobPart[], { type: opts.contentType ?? 'application/octet-stream' })
    if (cache) {
      try {
        await cache.put(
          key,
          new Response(blob, { status: 200, headers: { 'content-type': blob.type, 'content-length': String(loaded) } }),
        )
      } catch {
        // Quota or private mode: carry on without keeping a copy.
      }
    }
    return { data: await blob.arrayBuffer(), fromCache: false }
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
