import { useEffect, useRef } from 'react'
import { useTheme } from '@/hooks/use-theme'
import { cn } from '@/lib/utils'

export interface LevelSample {
  /** Audio time in seconds. */
  t: number
  /** 0..1 */
  v: number
}

export interface LevelTrack {
  samples: LevelSample[]
}

/** Map an RMS amplitude to 0..1 on a dB scale (−60 dBFS → 0, −6 dBFS → 1). */
export function levelFromRms(rms: number): number {
  const db = 20 * Math.log10(rms + 1e-7)
  return Math.max(0, Math.min(1, (db + 60) / 54))
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/**
 * Scrolling level history drawn on a canvas. Bars inside detected speech are
 * drawn in the accent colour over a shaded band; everything else is muted.
 * Reads from mutable refs so audio blocks never trigger React renders.
 */
export function LevelStrip({
  track,
  speech,
  windowSeconds = 12,
  active,
  className,
  label = 'Audio level',
}: {
  track: React.RefObject<LevelTrack>
  speech: React.RefObject<{ t0: number; t1: number | null }[]>
  windowSeconds?: number
  active: boolean
  className?: string
  label?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { resolved } = useTheme()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const colors = {
      speech: cssVar('--chart-1'),
      idle: cssVar('--chart-mute'),
      band: cssVar('--chart-1'),
      axis: cssVar('--chart-grid'),
    }
    let raf = 0
    let stopped = false

    const draw = () => {
      const dpr = Math.min(2, globalThis.devicePixelRatio || 1)
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      const samples = track.current?.samples ?? []
      const spans = speech.current ?? []
      const end = samples.length ? (samples[samples.length - 1] as LevelSample).t : 0
      const t1 = Math.max(windowSeconds, end)
      const t0 = t1 - windowSeconds
      const x = (t: number) => ((t - t0) / windowSeconds) * w
      const mid = h / 2

      ctx.fillStyle = colors.band
      ctx.globalAlpha = 0.1
      for (const s of spans) {
        const a = Math.max(t0, s.t0)
        const b = Math.min(t1, s.t1 ?? end)
        if (b > a) {
          ctx.beginPath()
          ctx.roundRect(x(a), 2, Math.max(1, x(b) - x(a)), h - 4, 4)
          ctx.fill()
        }
      }
      ctx.globalAlpha = 1
      ctx.fillStyle = colors.axis
      ctx.fillRect(0, mid, w, 1)

      // One bar per 3 px column; each takes the peak level of the samples it covers.
      const colW = 3
      const cols = Math.floor(w / colW)
      const peaks = new Float32Array(cols)
      for (let i = samples.length - 1; i >= 0; i--) {
        const s = samples[i] as LevelSample
        if (s.t < t0) break
        const c = Math.min(cols - 1, Math.floor(x(s.t) / colW))
        if (c >= 0 && s.v > (peaks[c] as number)) peaks[c] = s.v
      }
      const inSpeech = (t: number) => spans.some((s) => t >= s.t0 && t <= (s.t1 ?? Infinity))
      for (let c = 0; c < cols; c++) {
        const v = peaks[c] as number
        if (v <= 0) continue
        const t = t0 + ((c + 0.5) * colW * windowSeconds) / w
        const bar = Math.max(1.5, v * (h - 8))
        ctx.fillStyle = inSpeech(t) ? colors.speech : colors.idle
        ctx.beginPath()
        ctx.roundRect(c * colW, mid - bar / 2, colW - 1, bar, 1)
        ctx.fill()
      }
      if (!stopped && active) raf = requestAnimationFrame(draw)
    }
    draw()
    const ro = new ResizeObserver(() => {
      if (!active) draw()
    })
    ro.observe(canvas)
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [track, speech, windowSeconds, active, resolved])

  return <canvas ref={canvasRef} role="img" aria-label={label} className={cn('block h-16 w-full', className)} />
}
