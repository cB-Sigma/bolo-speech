const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const nf1 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 })

export const DASH = '–'

export function fmtMs(ms: number | null | undefined, unit = true): string {
  if (ms == null || !Number.isFinite(ms)) return DASH
  const v =
    Math.abs(ms) >= 10_000
      ? `${nf1.format(ms / 1000)}${unit ? ' s' : ''}`
      : `${nf0.format(Math.round(ms))}${unit ? ' ms' : ''}`
  return v
}

export function fmtInt(n: number | null | undefined): string {
  return n == null || !Number.isFinite(n) ? DASH : nf0.format(n)
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return DASH
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = bytes
  let i = 0
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000
    i++
  }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

export function fmtParams(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n <= 0) return DASH
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`
  if (n >= 1e3) return `${Math.round(n / 1e3)}K`
  return String(n)
}

/** Real-time factor: processing time ÷ audio duration. Lower is faster. */
export function fmtRtf(rtf: number | null | undefined): string {
  if (rtf == null || !Number.isFinite(rtf)) return DASH
  return rtf < 0.1 ? rtf.toFixed(3) : rtf.toFixed(2)
}

/** Speed as a multiple of real time (the inverse of RTF). */
export function fmtSpeed(rtf: number | null | undefined): string {
  if (rtf == null || !Number.isFinite(rtf) || rtf <= 0) return DASH
  const x = 1 / rtf
  return `${x >= 10 ? Math.round(x) : x.toFixed(1)}×`
}

export function fmtSeconds(s: number | null | undefined, digits = 1): string {
  if (s == null || !Number.isFinite(s)) return DASH
  return `${s.toFixed(digits)} s`
}

export function fmtDuration(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return DASH
  if (s < 60) return `${Math.floor(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`
}

export function fmtPercent(v: number | null | undefined, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return DASH
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtClock(d: Date | number | string): string {
  const date = d instanceof Date ? d : new Date(d)
  if (Number.isNaN(date.getTime())) return DASH
  return date.toLocaleTimeString('en-GB', { hour12: false })
}

let displayNames: Intl.DisplayNames | null | undefined

/** English name of a BCP-47 language code, falling back to the code itself. */
export function languageName(code: string): string {
  if (!code) return 'Any'
  if (code === 'multi') return 'Multilingual'
  if (displayNames === undefined) {
    try {
      displayNames = new Intl.DisplayNames(['en'], { type: 'language', fallback: 'none' })
    } catch {
      displayNames = null
    }
  }
  try {
    return displayNames?.of(code) ?? code
  } catch {
    return code
  }
}

/** A total amount of audio, in the largest unit that keeps it readable. */
export function fmtAudioTotal(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return DASH
  if (seconds < 90) return `${Math.round(seconds)} s`
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}
