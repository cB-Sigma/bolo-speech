import { useCallback, useEffect, useRef, useState } from 'react'
import { MicrophoneError, openBufferSource, openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { rms } from '@/lib/audio/pcm'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { diag } from '@/lib/diagnostics'
import type { MicFailure } from '@/lib/mic-help'
import { prefs } from '@/lib/prefs'
import { levelFromRms, type LevelTrack } from '@/components/charts/level-strip'
import { audioEngine } from '@/lib/browser-audio/engine'
import { BrowserTranscriber } from '@/lib/stt/browser'
import { DenoisingTranscriber } from '@/lib/stt/denoising'
import type { Denoise } from '@/lib/stt-options'
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
  /** Denoising on the server's live stream (`denoise` query parameter); left out, the server's default. */
  denoise?: Denoise
  /** Language for the in-browser model (`auto` or a code). */
  browserLanguage?: string
}

export type Phase = 'idle' | 'starting' | 'running' | 'finishing'
/** `requesting`: waiting for the permission prompt. `live`: audio is arriving. */
export type MicState = 'off' | 'requesting' | 'live'

export interface SessionError {
  kind: string
  message: string
  /** Present when the microphone was the problem. */
  mic?: MicFailure
}

const KEEP_SECONDS = 60
/** Audio captured while the recogniser is still getting ready is kept, up to this much. */
export const MAX_BUFFERED_SECONDS = 60

/**
 * Drives one transcription take: opens the audio source, fans each block out
 * to the server and/or in-browser recogniser, and keeps a level history and
 * the raw audio of the take for replay.
 *
 * The microphone is requested first, in the same turn as the click, and in
 * parallel with getting the recogniser ready. Whatever is said in the
 * meantime is buffered and handed over as soon as the recogniser can take it.
 */
export function useSttSession() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [server, setServer] = useState<Transcriber | null>(null)
  const [browser, setBrowser] = useState<Transcriber | null>(null)
  /** The denoiser in front of the in-browser recogniser, when one is chosen. */
  const [denoiser, setDenoiser] = useState<DenoisingTranscriber | null>(null)
  const [input, setInput] = useState<SessionInput | null>(null)
  const [error, setError] = useState<SessionError | null>(null)
  const [mic, setMic] = useState<MicState>('off')
  /** Seconds of speech being held until the recogniser is ready; null when nothing is held. */
  const [buffered, setBuffered] = useState<number | null>(null)
  const levels = useRef<LevelTrack>({ samples: [] })
  const taken = useRef<Float32Array[]>([])
  const source = useRef<AudioSource | null>(null)
  const active = useRef<Transcriber[]>([])
  const run = useRef(0)
  const phaseRef = useRef<Phase>('idle')

  const phaseTo = useCallback((p: Phase) => {
    phaseRef.current = p
    setPhase(p)
  }, [])

  const cancel = useCallback(() => {
    run.current += 1
    void source.current?.stop()
    source.current = null
    active.current.forEach((t) => t.cancel())
    active.current = []
    setMic('off')
    setBuffered(null)
    phaseTo('idle')
  }, [phaseTo])

  const stop = useCallback(async () => {
    // Stopping before the recogniser is ready abandons the take.
    if (phaseRef.current === 'starting') {
      cancel()
      return
    }
    const id = run.current
    const src = source.current
    source.current = null
    if (!src && active.current.length === 0) return
    phaseTo('finishing')
    await src?.stop()
    setMic('off')
    await Promise.all(active.current.map((t) => t.finish().catch(() => undefined)))
    if (run.current === id) phaseTo('idle')
  }, [cancel, phaseTo])

  const start = useCallback(
    (inp: SessionInput, cfg: SessionConfig): void => {
      cancel()
      const id = ++run.current
      setError(null)
      setInput(inp)
      phaseTo('starting')
      levels.current.samples = []
      taken.current = []

      // Blocks go to the recognisers once they are ready; until then they wait here.
      let targets: Transcriber[] | null = null
      let held: Float32Array[] = []
      let heldSamples = 0
      let lastHeldReport = 0
      let seconds = 0
      let kept = 0
      const onBlock = (block: Float32Array) => {
        if (run.current !== id) return
        seconds += block.length / STT_RATE
        const samples = levels.current.samples
        samples.push({ t: seconds, v: levelFromRms(rms(block)) })
        if (samples.length > 6000) samples.splice(0, 1000)
        if (kept < KEEP_SECONDS * STT_RATE) {
          taken.current.push(block.slice())
          kept += block.length
        }
        if (targets) {
          for (const t of targets) t.push(block)
          return
        }
        held.push(block.slice())
        heldSamples += block.length
        while (heldSamples > MAX_BUFFERED_SECONDS * STT_RATE && held.length > 1) {
          heldSamples -= (held.shift() as Float32Array).length
        }
        const now = performance.now()
        if (now - lastHeldReport > 250) {
          lastHeldReport = now
          setBuffered(heldSamples / STT_RATE)
        }
      }

      // 1. The microphone, before anything asynchronous: the permission prompt must come from this click.
      const micPromise = inp.kind === 'mic' ? openMicrophone(onBlock) : null
      if (micPromise) {
        setMic('requesting')
        diag.event('microphone', 'requested')
      }

      // 2. The recognisers, in parallel.
      const srv =
        cfg.mode !== 'browser'
          ? new ServerTranscriber({
              model: cfg.model,
              language: cfg.language,
              telephony: cfg.telephony,
              denoise: cfg.denoise,
            })
          : null
      const p = prefs.get()
      const recogniser =
        cfg.mode !== 'server'
          ? new BrowserTranscriber(sharedEngine(), {
              language: cfg.browserLanguage ?? 'auto',
              maxRepeats: p.browserMaxRepeats,
            })
          : null
      const dn =
        recogniser && p.browserDenoise
          ? new DenoisingTranscriber(recogniser, audioEngine('denoise'), {
              model: p.browserDenoise,
              strength: p.browserDenoiseStrength,
              device: p.browserAudioDevice,
            })
          : null
      setDenoiser(dn)
      const brw: Transcriber | null = dn ?? recogniser
      const all = [srv, brw].filter((t): t is Transcriber => t != null)
      active.current = all
      setServer(srv)
      setBrowser(brw)
      const starts = Promise.allSettled(all.map((t) => t.start()))

      const failed = (err: SessionError) => {
        if (run.current !== id) return
        run.current += 1
        void source.current?.stop()
        source.current = null
        all.forEach((t) => t.cancel())
        active.current = []
        setMic('off')
        setBuffered(null)
        setError(err)
        phaseTo('idle')
      }

      void (async () => {
        if (micPromise) {
          try {
            const src = await micPromise
            if (run.current !== id) {
              void src.stop()
              return
            }
            source.current = src
            setMic('live')
            setBuffered(0)
            diag.event('microphone', 'open')
          } catch (err) {
            const mic = err instanceof MicrophoneError ? err.failure : undefined
            diag.error(
              'microphone',
              mic
                ? `${mic.kind}: ${mic.name || 'error'}: ${mic.detail} (after ${Math.round(mic.elapsedMs)} ms, permission ${mic.permission})`
                : String(err),
            )
            failed({
              kind: err instanceof MicrophoneError ? err.kind : 'failed',
              message: err instanceof Error ? err.message : 'Could not open the microphone',
              mic,
            })
            return
          }
        }

        const results = await starts
        if (run.current !== id) return
        const usable = all.filter((_, i) => results[i]?.status === 'fulfilled')
        if (usable.length === 0) {
          const first = results.find((r): r is PromiseRejectedResult => r.status === 'rejected')
          failed({ kind: 'engine', message: first?.reason instanceof Error ? first.reason.message : 'Could not start' })
          return
        }
        active.current = usable

        // Hand over what was said while the recogniser was getting ready, then go live.
        const backlog = held
        held = []
        heldSamples = 0
        for (const block of backlog) for (const t of usable) t.push(block)
        targets = usable
        setBuffered(null)
        if (backlog.length > 0) diag.event('microphone', `handed over ${backlog.length} buffered blocks`)

        if (inp.kind === 'file') {
          source.current = openBufferSource(inp.samples ?? new Float32Array(0), inp.sampleRate ?? STT_RATE, onBlock, {
            onDone: () => {
              if (run.current === id) void stop()
            },
          })
        }
        phaseTo('running')
      })()
    },
    [cancel, stop, phaseTo],
  )

  const clearError = useCallback(() => setError(null), [])

  useEffect(() => cancel, [cancel])

  return {
    phase,
    server,
    browser,
    denoiser,
    input,
    error,
    mic,
    buffered,
    levels,
    taken,
    start,
    stop,
    cancel,
    clearError,
  }
}
