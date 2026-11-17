import { Loader2, Mic, Scale, Square, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { describeError, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import { encodeWav } from '@/lib/audio/wav'
import { fmtSeconds } from '@/lib/format'
import { CALIBRATED_MODEL, cosine, SAME_SPEAKER, SIMILAR, verdictFor } from '@/lib/similarity'
import { cn } from '@/lib/utils'

interface Clip {
  blob: Blob
  name: string
  seconds: number | null
}

const DEFAULT = '__default'

/** One of the two recordings: upload a file or record one here. */
function ClipSlot({
  label,
  clip,
  onChange,
  disabled,
  testId,
}: {
  label: string
  clip: Clip | null
  onChange: (c: Clip | null) => void
  disabled?: boolean
  testId: string
}) {
  const input = useRef<HTMLInputElement>(null)
  const mic = useRef<AudioSource | null>(null)
  const blocks = useRef<Float32Array[]>([])
  const [recording, setRecording] = useState(false)

  useEffect(() => () => void mic.current?.stop(), [])

  const record = async () => {
    if (recording) {
      const m = mic.current
      mic.current = null
      setRecording(false)
      await m?.stop()
      const audio = concatFloat(blocks.current)
      blocks.current = []
      const seconds = audio.length / STT_RATE
      if (seconds < 0.5) {
        toast.error('The recording is too short', { description: 'Record at least a second or two of speech.' })
        return
      }
      onChange({
        blob: new Blob([encodeWav(audio, STT_RATE) as BlobPart], { type: 'audio/wav' }),
        name: 'recording.wav',
        seconds,
      })
      return
    }
    blocks.current = []
    setRecording(true)
    try {
      mic.current = await openMicrophone((b) => blocks.current.push(b.slice()))
    } catch (err) {
      setRecording(false)
      toast.error('Could not open the microphone', { description: err instanceof Error ? err.message : undefined })
    }
  }

  return (
    <div className="grid min-w-0 gap-1.5 rounded-lg border border-border p-2.5" data-testid={testId}>
      <span className="text-xs font-medium">{label}</span>
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" variant="outline" onClick={() => input.current?.click()} disabled={disabled || recording}>
          <Upload /> File
        </Button>
        <input
          ref={input}
          type="file"
          accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
          className="sr-only"
          tabIndex={-1}
          aria-label={`${label}: recording`}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onChange({ blob: f, name: f.name, seconds: null })
            e.target.value = ''
          }}
        />
        <Button
          size="sm"
          variant={recording ? 'destructive' : 'outline'}
          onClick={() => void record()}
          disabled={disabled}
        >
          {recording ? <Square className="fill-current" /> : <Mic />} {recording ? 'Stop' : 'Record'}
        </Button>
      </div>
      {clip ? (
        <span className="flex min-w-0 items-center gap-1 text-xs text-muted-foreground">
          <span className="truncate">{clip.name}</span>
          {clip.seconds != null && <span className="num">· {fmtSeconds(clip.seconds)}</span>}
          <Button
            size="icon-sm"
            variant="ghost"
            className="ml-auto size-6"
            onClick={() => onChange(null)}
            aria-label={`Remove ${label.toLowerCase()}`}
          >
            <X />
          </Button>
        </span>
      ) : (
        <span className="text-xs text-muted-foreground">{recording ? 'Recording…' : 'No recording yet'}</span>
      )}
    </div>
  )
}

/**
 * Compare two voices on the server: the speaker embedding of each recording (`POST /v1/audio/speech/embedding`)
 * and their cosine. With the default embedding model, 0.50 and above is what the same person scores.
 */
export function VoiceCompare({ className }: { className?: string }) {
  const diarizers = useModels('diarization').models.filter((m) => m.installed && m.available !== false)
  const [a, setA] = useState<Clip | null>(null)
  const [b, setB] = useState<Clip | null>(null)
  const [model, setModel] = useState(DEFAULT)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ score: number; model: string; dims: number } | null>(null)

  const change = (set: (c: Clip | null) => void) => (c: Clip | null) => {
    set(c)
    setResult(null)
  }

  const compare = async () => {
    if (!a || !b) return
    setBusy(true)
    setResult(null)
    try {
      const chosen = model === DEFAULT ? undefined : model
      const [ea, eb] = await Promise.all([
        api.speakerEmbedding(a.blob, a.name, chosen),
        api.speakerEmbedding(b.blob, b.name, chosen),
      ])
      const va = ea.data[0]?.embedding ?? []
      const vb = eb.data[0]?.embedding ?? []
      const score = cosine(va, vb)
      if (score == null) throw new Error('The server returned embeddings that cannot be compared.')
      setResult({ score, model: ea.model, dims: va.length })
    } catch (err) {
      toast.error('Could not compare the voices', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  const calibrated = result?.model === CALIBRATED_MODEL
  const verdict = result ? verdictFor(result.score) : null

  return (
    <Panel
      title="Compare two voices"
      description="Speaker embeddings on the server, and how alike they are"
      className={className}
      data-testid="voice-compare"
    >
      <div className="grid gap-3">
        <div className="grid gap-2 sm:grid-cols-2">
          <ClipSlot label="First voice" clip={a} onChange={change(setA)} disabled={busy} testId="compare-a" />
          <ClipSlot label="Second voice" clip={b} onChange={change(setB)} disabled={busy} testId="compare-b" />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select value={model} onValueChange={setModel} disabled={busy}>
            <SelectTrigger className="h-8 w-56" aria-label="Embedding model" data-testid="compare-model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              <SelectItem value={DEFAULT}>Embedding: server default</SelectItem>
              {diarizers.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  <span className="num">{m.id}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" onClick={() => void compare()} disabled={!a || !b || busy} data-testid="compare-run">
            {busy ? <Loader2 className="animate-spin" /> : <Scale />} Compare
          </Button>
        </div>
        {result && verdict && (
          <div className="grid gap-1" data-testid="compare-result">
            <p className="flex items-baseline gap-2">
              <span className="num text-3xl leading-none font-semibold tracking-tight">{result.score.toFixed(2)}</span>
              <span
                className={cn(
                  'text-sm',
                  !calibrated
                    ? 'text-muted-foreground'
                    : verdict === 'same speaker'
                      ? 'text-good-text'
                      : verdict === 'similar'
                        ? 'text-warning-text'
                        : 'text-muted-foreground',
                )}
              >
                {calibrated ? verdict : 'cosine similarity'}
              </span>
            </p>
            <p className="text-xs leading-4 text-muted-foreground">
              <span className="num">{result.model}</span> · {result.dims} numbers per voice.{' '}
              {calibrated
                ? `With this model the same person scored 0.45 to 0.93 and different people −0.30 to 0.53; ${SAME_SPEAKER.toFixed(2)} and above reads as the same speaker, ${SIMILAR.toFixed(2)} and above as similar.`
                : `The thresholds bolo uses (${SAME_SPEAKER.toFixed(2)} for the same speaker) were measured with ${CALIBRATED_MODEL}; another model has another scale.`}
            </p>
          </div>
        )}
        <p className="text-xs leading-4 text-muted-foreground">
          A clip needs half a second of speech at least; the first 30 s are used. A score is a likeness, not proof of
          identity.
        </p>
      </div>
    </Panel>
  )
}
