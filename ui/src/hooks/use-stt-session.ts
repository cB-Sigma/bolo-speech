import { useCallback, useEffect, useRef, useState } from 'react'
import { MicrophoneError, openBufferSource, openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { rms } from '@/lib/audio/pcm'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { levelFromRms, type LevelTrack } from '@/components/charts/level-strip'
import { BrowserTranscriber } from '@/lib/stt/browser'
import { ServerTranscriber } from '@/lib/stt/server'
import type { Transcriber } from '@/lib/stt/transcriber'

export type EngineMode = 'server' | 'browser' | 'compare'
export type SourceKind = 'mic' | 'file'

export interface SessionInput {
  kind: SourceKind
  label: string
  /** For `file`: decoded mono audio. */
  samples?: Float32Array
  sampleRate?: number
}

export interface SessionConfig {
  mode: EngineMode
  model: string
  language: string
  telephony: boolean
}

export type Phase = 'idle' | 'starting' | 'running' | 'finishing'

const KEEP_SECONDS = 60

/**
 * Drives one transcription take: opens the audio source, fans each block out
 * to the server and/or in-browser recogniser, and keeps a level history and
 * the raw audio of the take for replay.
 */
export function useSttSession() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [server, setServer] = useState<Transcriber | null>(null)
  const [browser, setBrowser] = useState<Transcriber | null>(null)
  const [input, setInput] = useState<SessionInput | null>(null)
  const [error, setError] = useState<{ kind: string; message: string } | null>(null)
  const levels = useRef<LevelTrack>({ samples: [] })
  const taken = useRef<Float32Array[]>([])
  const source = useRef<AudioSource | null>(null)
  const active = useRef<Transcriber[]>([])
  const run = useRef(0)

  const stop = useCallback(async () => {
    const id = run.current
    const src = source.current
    source.current = null
    if (!src && active.current.length === 0) return
    setPhase('finishing')
    await src?.stop()
    await Promise.all(active.current.map((t) => t.finish().catch(() => undefined)))
    if (run.current === id) setPhase('idle')
  }, [])

  const cancel = useCallback(() => {
    run.current += 1
    void source.current?.stop()
    source.current = null
    active.current.forEach((t) => t.cancel())
    active.current = []
    setPhase('idle')
  }, [])

  const start = useCallback(
    async (inp: SessionInput, cfg: SessionConfig) => {
      cancel()
      const id = ++run.current
      setError(null)
      setInput(inp)
      setPhase('starting')
      levels.current.samples = []
      taken.current = []
      const srv =
        cfg.mode !== 'browser'
          ? new ServerTranscriber({ model: cfg.model, language: cfg.language, telephony: cfg.telephony })
          : null
      const brw = cfg.mode !== 'server' ? new BrowserTranscriber(sharedEngine()) : null
      const all = [srv, brw].filter((t): t is ServerTranscriber | BrowserTranscriber => t != null)
      active.current = all
      setServer(srv)
      setBrowser(brw)

      const results = await Promise.allSettled(all.map((t) => t.start()))
      if (run.current !== id) return
      const usable = all.filter((_, i) => results[i]?.status === 'fulfilled')
      if (usable.length === 0) {
        const first = results.find((r): r is PromiseRejectedResult => r.status === 'rejected')
        setError({ kind: 'engine', message: first?.reason instanceof Error ? first.reason.message : 'Could not start' })
        setPhase('idle')
        return
      }
      active.current = usable

      let seconds = 0
      let kept = 0
      const onBlock = (block: Float32Array) => {
        if (run.current !== id) return
        for (const t of usable) t.push(block)
        seconds += block.length / STT_RATE
        const samples = levels.current.samples
        samples.push({ t: seconds, v: levelFromRms(rms(block)) })
        if (samples.length > 6000) samples.splice(0, 1000)
        if (kept < KEEP_SECONDS * STT_RATE) {
          taken.current.push(block.slice())
          kept += block.length
        }
      }
      try {
        if (inp.kind === 'mic') {
          source.current = await openMicrophone(onBlock)
        } else {
          source.current = openBufferSource(inp.samples ?? new Float32Array(0), inp.sampleRate ?? STT_RATE, onBlock, {
            onDone: () => {
              if (run.current === id) void stop()
            },
          })
        }
        if (run.current !== id) {
          void source.current.stop()
          return
        }
        setPhase('running')
      } catch (err) {
        usable.forEach((t) => t.cancel())
        active.current = []
        setError({
          kind: err instanceof MicrophoneError ? err.kind : 'failed',
          message: err instanceof Error ? err.message : 'Could not open the audio source',
        })
        setPhase('idle')
      }
    },
    [cancel, stop],
  )

  useEffect(() => cancel, [cancel])

  return { phase, server, browser, input, error, levels, taken, start, stop, cancel }
}
