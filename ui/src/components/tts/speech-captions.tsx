import { Captions, Loader2, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useClipPlayer } from '@/components/align/align-panel'
import { AlignedWords } from '@/components/align/aligned-words'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { describeError } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { api } from '@/lib/api/client'
import type { Alignment } from '@/lib/api/types'
import { encodeWav } from '@/lib/audio/wav'
import type { TtsSession } from '@/lib/tts/session'

type Captioning =
  { status: 'waiting' | 'aligning' } | { status: 'done'; result: Alignment } | { status: 'error'; error: string }

/**
 * The synthesised speech with every word highlighted as it is spoken. When synthesis is done, its audio and the text
 * go to the server's forced aligner (`POST /v1/audio/alignments`); the words then follow the live playback, and
 * replay (or a click on a word) plays the audio again with them.
 */
export function SpeechCaptions({
  session,
  text,
  language,
  model,
}: {
  session: TtsSession
  text: string
  language?: string
  /** An alignment model; the server's default when empty. */
  model?: string
}) {
  const state = useStore(session.store)
  // the answer for one audio and text; anything else is still waiting or being aligned
  const [answer, setAnswer] = useState<{ wav: Blob; text: string; result?: Alignment; error?: string } | null>(null)
  const synthesised = state.totalMs != null && state.sampleRate != null && state.status !== 'error'
  const wav = useMemo(
    () =>
      synthesised && state.sampleRate
        ? new Blob([encodeWav(session.audio(), state.sampleRate) as BlobPart], { type: 'audio/wav' })
        : null,
    [synthesised, session, state.sampleRate],
  )

  useEffect(() => {
    if (!wav) return
    const abort = new AbortController()
    api
      .align(wav, 'speech.wav', text, { model: model || undefined, language: language || undefined }, abort.signal)
      .then((result) => setAnswer({ wav, text, result }))
      .catch((err: unknown) => {
        if (!abort.signal.aborted) setAnswer({ wav, text, error: describeError(err) })
      })
    return () => abort.abort()
  }, [wav, text, language, model])

  const captioning: Captioning = !wav
    ? { status: 'waiting' }
    : answer?.wav !== wav || answer.text !== text
      ? { status: 'aligning' }
      : answer.result
        ? { status: 'done', result: answer.result }
        : { status: 'error', error: answer.error ?? 'failed' }

  // while the first playback runs, follow it; afterwards, our own player
  const replay = useClipPlayer(captioning.status === 'done' ? wav : null)
  const [live, setLive] = useState<number | null>(null)
  const playing = state.status === 'playing'
  useEffect(() => {
    if (!playing || !session.player) return
    let frame = 0
    const tick = () => {
      setLive(session.player?.position ?? null)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, session])

  const time = replay.playing || replay.time != null ? replay.time : playing ? live : null
  return (
    <Panel
      title="Word captions"
      description="Each word highlighted as it is spoken, from the server's forced aligner"
      data-testid="speech-captions"
      actions={
        captioning.status === 'done' ? (
          <Button size="sm" variant="secondary" onClick={replay.toggle} data-testid="captions-replay">
            <Play className="fill-current" /> {replay.playing ? 'Pause' : 'Play with words'}
          </Button>
        ) : null
      }
    >
      {captioning.status === 'waiting' && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Captions className="size-4" aria-hidden /> The words are aligned once synthesis has finished.
        </p>
      )}
      {captioning.status === 'aligning' && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden /> Aligning the text to the audio…
        </p>
      )}
      {captioning.status === 'error' && (
        <p className="text-xs text-critical" data-testid="captions-error">
          Could not align the words: {captioning.error}
        </p>
      )}
      {captioning.status === 'done' && (
        <>
          <AlignedWords
            words={captioning.result.words}
            time={time}
            onSeek={replay.seek}
            lang={language}
            testId="caption-words"
          />
          <p className="mt-1 text-xs leading-4 text-muted-foreground">
            <span className="num">{captioning.result.model}</span> · the text as typed, aligned to the audio. Numbers
            and abbreviations the voice read out in words are matched as written.
          </p>
        </>
      )}
    </Panel>
  )
}
