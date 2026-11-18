import { Download, FileAudio, Loader2, Pause, Play, TextCursorInput } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { AlignedWords, wordsToVtt } from '@/components/align/aligned-words'
import { CatalogSelect } from '@/components/catalog-select'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { Alignment, Sample, ServerInfo } from '@/lib/api/types'
import { download } from '@/lib/bench'
import { autoPull } from '@/lib/capabilities'
import { fmtSeconds } from '@/lib/format'

/** Plays a clip and reports where playback is, many times a second while it plays. */
export function useClipPlayer(blob: Blob | null) {
  const audio = useRef<HTMLAudioElement | null>(null)
  const [time, setTime] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const el = new Audio(url)
    audio.current = el
    let frame = 0
    const tick = () => {
      setTime(el.currentTime)
      if (!el.paused) frame = requestAnimationFrame(tick)
    }
    el.onplay = () => {
      setPlaying(true)
      frame = requestAnimationFrame(tick)
    }
    el.onpause = () => {
      setPlaying(false)
      cancelAnimationFrame(frame)
    }
    el.onended = () => {
      setPlaying(false)
      setTime(null)
    }
    return () => {
      cancelAnimationFrame(frame)
      el.pause()
      audio.current = null
      URL.revokeObjectURL(url)
      setTime(null)
      setPlaying(false)
    }
  }, [blob])
  return {
    time,
    playing,
    toggle: () => {
      const el = audio.current
      if (!el) return
      if (el.paused) void el.play().catch(() => undefined)
      else el.pause()
    },
    seek: (seconds: number) => {
      const el = audio.current
      if (!el) return
      el.currentTime = seconds
      setTime(seconds)
      if (el.paused) void el.play().catch(() => undefined)
    },
  }
}

/**
 * "Align text to audio": a recording and what is said in it, sent to the server's forced aligner
 * (`POST /v1/audio/alignments`); the words come back with their times and play highlighted.
 */
export function AlignPanel({
  info,
  samples,
  initialText,
  initialClip,
  className,
}: {
  info: ServerInfo | undefined
  samples: Sample[]
  /** The transcript to start from (the page's last transcript). */
  initialText?: string
  /** The recording to start from (the page's last file or sample). */
  initialClip?: Clip | null
  className?: string
}) {
  const models = useModels('align').models
  // the page's latest recording and transcript, until the visitor picks their own
  const [ownClip, setOwnClip] = useState<Clip | null | undefined>(undefined)
  const [ownText, setOwnText] = useState<string | null>(null)
  const clip = ownClip !== undefined ? ownClip : (initialClip ?? null)
  const text = ownText ?? initialText ?? ''
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<{ alignment: Alignment; clip: Clip; text: string } | null>(null)
  // an answer is shown while the recording and the text are the ones it was for
  const result = answer && answer.clip === clip && answer.text === text ? answer.alignment : null
  const player = useClipPlayer(result ? (clip?.blob ?? null) : null)
  const audioSamples = samples.filter((s) => s.kind === 'audio' && s.url && s.id !== 'tone-bursts')

  const pickSample = async (s: Sample) => {
    if (!s.url) return
    try {
      const bytes = await api.bytes(s.url)
      setOwnClip({ blob: new Blob([bytes], { type: 'audio/wav' }), name: `${s.id}.wav`, seconds: s.seconds ?? null })
      if (s.text) setOwnText(s.text)
    } catch (err) {
      toast.error('Could not load the sample', { description: describeError(err) })
    }
  }

  const run = async () => {
    if (!clip || !text.trim()) return
    setBusy(true)
    setAnswer(null)
    try {
      const alignment = await api.align(clip.blob, clip.name, text, { model: model || undefined })
      setAnswer({ alignment, clip, text })
    } catch (err) {
      toast.error('Could not align the text', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="Align text to audio"
      description="Word timestamps for a recording and what is said in it (forced alignment)"
      className={className}
      data-testid="align-panel"
    >
      <div className="grid gap-3">
        <ClipSlot label="Recording" clip={clip} onChange={setOwnClip} disabled={busy} testId="align-clip" />
        {audioSamples.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-muted-foreground">Samples</span>
            {audioSamples.map((s) => (
              <Button
                key={s.id}
                size="sm"
                variant={clip?.name === `${s.id}.wav` ? 'secondary' : 'ghost'}
                className="border border-border"
                onClick={() => void pickSample(s)}
                disabled={busy}
                data-testid={`align-sample-${s.id}`}
              >
                <FileAudio /> <span className="num text-xs">{s.id}</span>
              </Button>
            ))}
          </div>
        )}
        <Textarea
          value={text}
          onChange={(e) => setOwnText(e.target.value)}
          placeholder="What is said in the recording"
          aria-label="Text to align"
          rows={3}
          disabled={busy}
          data-testid="align-text"
        />
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Alignment model"
            defaultLabel="Server default"
            disabled={busy}
            autoPull={autoPull(info)}
            extra={(m) => (m.commercial_use === false ? `${m.license}, non-commercial` : m.license)}
            className="w-64 max-sm:w-full"
            testId="align-model"
          />
          <Button size="sm" onClick={() => void run()} disabled={!clip || !text.trim() || busy} data-testid="align-run">
            {busy ? <Loader2 className="animate-spin" /> : <TextCursorInput />} Align
          </Button>
        </div>
        {result && (
          <div className="grid gap-2" data-testid="align-result">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary" onClick={player.toggle} data-testid="align-play">
                {player.playing ? <Pause /> : <Play className="fill-current" />} {player.playing ? 'Pause' : 'Play'}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => download('words.vtt', wordsToVtt(result.words), 'text/vtt')}
                data-testid="align-vtt"
              >
                <Download /> WebVTT
              </Button>
              <span className="num text-xs text-muted-foreground">
                {result.model} · {result.words.length} words · {fmtSeconds(result.duration)}
              </span>
            </div>
            <AlignedWords words={result.words} time={player.time} onSeek={player.seek} />
            <p className="text-xs leading-4 text-muted-foreground">
              Click a word to play from it. A dotted underline marks a word the aligner was unsure of (score below 50
              %): it may not be in the recording.
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}
