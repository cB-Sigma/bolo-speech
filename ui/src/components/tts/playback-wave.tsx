import { useEffect, useRef } from 'react'
import { useTheme } from '@/hooks/use-theme'
import type { TtsSession } from '@/lib/tts/session'
import { cn } from '@/lib/utils'

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/**
 * Overview of the synthesised audio that fills in as chunks arrive, with the
 * played part in the accent colour and thin rules at chunk boundaries.
 */
export function PlaybackWave({ session, className }: { session: TtsSession | null; className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { resolved } = useTheme()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const colors = {
      played: cssVar('--chart-1'),
      ahead: cssVar('--chart-mute'),
      axis: cssVar('--chart-grid'),
      mark: cssVar('--foreground'),
    }
    let raf = 0
    let stopped = false
    let peaks = new Float32Array(0)
    let peaksFor = -1
    let peaksWidth = 0

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
      const mid = h / 2
      ctx.fillStyle = colors.axis
      ctx.fillRect(0, mid, w, 1)
      const state = session?.store.get()
      if (session && state && state.audioSeconds > 0) {
        const colW = 3
        const cols = Math.floor(w / colW)
        if (peaksFor !== state.audioSeconds || peaksWidth !== cols) {
          const audio = session.audio()
          peaks = new Float32Array(cols)
          const per = audio.length / cols
          for (let c = 0; c < cols; c++) {
            let p = 0
            const end = Math.min(audio.length, Math.floor((c + 1) * per))
            for (let i = Math.floor(c * per); i < end; i += 4) {
              const a = Math.abs(audio[i] as number)
              if (a > p) p = a
            }
            peaks[c] = p
          }
          peaksFor = state.audioSeconds
          peaksWidth = cols
        }
        const pos = session.player ? session.player.position / state.audioSeconds : 0
        const playing = state.status === 'playing' || state.status === 'synthesizing'
        const playedCols = state.status === 'done' ? cols : Math.floor(pos * cols)
        for (let c = 0; c < cols; c++) {
          const bar = Math.max(1.5, Math.min(1, (peaks[c] as number) * 2.4) * (h - 8))
          ctx.fillStyle = c <= playedCols ? colors.played : colors.ahead
          ctx.beginPath()
          ctx.roundRect(c * colW, mid - bar / 2, colW - 1, bar, 1)
          ctx.fill()
        }
        ctx.fillStyle = colors.mark
        ctx.globalAlpha = 0.35
        for (const ch of state.chunks) {
          if (ch.t0 > 0) ctx.fillRect(Math.round((ch.t0 / state.audioSeconds) * w), 2, 1, h - 4)
        }
        ctx.globalAlpha = 1
        if (playing && pos > 0 && pos < 1) ctx.fillRect(Math.round(pos * w), 0, 1.5, h)
      }
      if (!stopped) raf = requestAnimationFrame(draw)
    }
    draw()
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
    }
  }, [session, resolved])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Waveform of the synthesised audio"
      className={cn('block h-20 w-full', className)}
    />
  )
}
