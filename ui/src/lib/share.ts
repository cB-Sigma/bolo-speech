/**
 * "Share this result": a latency summary packed into the URL, so a link
 * carries the numbers without any backend. The payload is tiny, versioned and
 * validated on the way back in, because anyone can edit a URL.
 */
export interface SharedResult {
  /** Model label. */
  m: string
  /** Where it ran, e.g. "WASM, 4 threads". */
  d: string
  /** Utterances measured. */
  n: number
  /** First partial p50, final p50, final p95 (ms). */
  fp: number | null
  f50: number
  f95: number
  /** Real-time factor. */
  rtf: number | null
}

const num = (v: unknown, max: number): number | null =>
  typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= max ? v : null

export function encodeShare(r: SharedResult): string {
  const compact = [
    1,
    r.m.slice(0, 40),
    r.d.slice(0, 40),
    r.n,
    r.fp == null ? null : Math.round(r.fp),
    Math.round(r.f50),
    Math.round(r.f95),
    r.rtf == null ? null : Math.round(r.rtf * 1000) / 1000,
  ]
  return btoa(unescape(encodeURIComponent(JSON.stringify(compact))))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')
}

export function decodeShare(token: string | null | undefined): SharedResult | null {
  if (!token || token.length > 400) return null
  try {
    const raw = JSON.parse(decodeURIComponent(escape(atob(token.replace(/-/g, '+').replace(/_/g, '/'))))) as unknown
    if (!Array.isArray(raw) || raw[0] !== 1 || raw.length !== 8) return null
    const [, m, d, n, fp, f50, f95, rtf] = raw as unknown[]
    const n2 = num(n, 10_000)
    const a = num(f50, 600_000)
    const b = num(f95, 600_000)
    if (typeof m !== 'string' || typeof d !== 'string' || n2 == null || n2 < 1 || a == null || b == null) return null
    return {
      m: m.slice(0, 40),
      d: d.slice(0, 40),
      n: Math.round(n2),
      fp: num(fp, 600_000),
      f50: a,
      f95: b,
      rtf: num(rtf, 1000),
    }
  } catch {
    return null
  }
}

/** Link to the playground carrying the result (hash routing: the query lives inside the hash). */
export function shareUrl(r: SharedResult, loc: Pick<Location, 'origin' | 'pathname'> = globalThis.location): string {
  return `${loc.origin}${loc.pathname}#/stt?r=${encodeShare(r)}`
}
