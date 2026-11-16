import { BookmarkPlus, Mic, Square, Trash2, Upload, Volume2 } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import { encodeWav } from '@/lib/audio/wav'
import { fmtSeconds } from '@/lib/format'

/** The server uses up to 30 s of the reference and refuses less than half a second. */
export const MAX_REFERENCE_S = 30
export const MIN_REFERENCE_S = 0.5

export interface CloneReference {
  /** `data:audio/…;base64,…` */
  dataUrl: string
  label: string
  seconds: number | null
  text: string
}

export function bytesToDataUrl(bytes: Uint8Array, type: string): string {
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return `data:${type || 'audio/wav'};base64,${btoa(bin)}`
}

/**
 * Reference recording for models that clone voices: upload a file or record
 * one here (the microphone is asked for at the tap), plus what is said in it.
 * With `onSave` (a server that keeps saved voices) the reference can be kept
 * as a saved voice.
 */
export function VoiceClonePanel({
  enabled,
  onEnabledChange,
  reference,
  onReferenceChange,
  onSave,
  disabled,
  className,
}: {
  enabled: boolean
  onEnabledChange: (on: boolean) => void
  reference: CloneReference | null
  onReferenceChange: (r: CloneReference | null) => void
  /** Keep the reference as a saved voice (opens the save dialog). */
  onSave?: (r: CloneReference) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const fileInput = useRef<HTMLInputElement>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const mic = useRef<AudioSource | null>(null)
  const blocks = useRef<Float32Array[]>([])
  const wanted = useRef(false)
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined)

  useEffect(
    () => () => {
      clearInterval(timer.current)
      void mic.current?.stop()
    },
    [],
  )

  const finishRecording = async () => {
    wanted.current = false
    clearInterval(timer.current)
    const m = mic.current
    mic.current = null
    setRecording(false)
    await m?.stop()
    const audio = concatFloat(blocks.current)
    blocks.current = []
    const seconds = audio.length / STT_RATE
    if (seconds < MIN_REFERENCE_S) {
      toast.error('The recording is too short', { description: 'Record at least a few seconds of clear speech.' })
      return
    }
    onReferenceChange({
      dataUrl: bytesToDataUrl(encodeWav(audio, STT_RATE), 'audio/wav'),
      label: 'Recorded here',
      seconds,
      text: reference?.text ?? '',
    })
  }

  const record = () => {
    // The permission prompt must come straight from the tap (iOS Safari).
    blocks.current = []
    wanted.current = true
    const opening = openMicrophone((block) => {
      blocks.current.push(block)
    })
    setRecording(true)
    setElapsed(0)
    const started = performance.now()
    timer.current = setInterval(() => {
      const s = (performance.now() - started) / 1000
      setElapsed(s)
      if (s >= MAX_REFERENCE_S) void finishRecording()
    }, 200)
    opening.then(
      (m) => {
        // Stopped before the microphone was ready: close it again.
        if (wanted.current) mic.current = m
        else void m.stop()
      },
      (err: unknown) => {
        wanted.current = false
        clearInterval(timer.current)
        setRecording(false)
        toast.error('Could not open the microphone', { description: err instanceof Error ? err.message : undefined })
      },
    )
  }

  const onFile = async (f: File) => {
    if (f.size > 20 * 1024 * 1024) {
      toast.error('That file is too large', { description: 'Use a clip of up to 30 seconds.' })
      return
    }
    const bytes = new Uint8Array(await f.arrayBuffer())
    onReferenceChange({
      dataUrl: bytesToDataUrl(bytes, f.type),
      label: f.name,
      seconds: null,
      text: reference?.text ?? '',
    })
  }

  const play = () => {
    if (!reference) return
    const a = new Audio(reference.dataUrl)
    void a.play().catch(() => toast.error('This browser cannot play that file; the server may still read it.'))
  }

  return (
    <Panel
      title="Clone a voice"
      description="This model can imitate a voice from a short recording"
      className={className}
      data-testid="voice-clone"
      actions={
        <Switch
          checked={enabled}
          onCheckedChange={onEnabledChange}
          disabled={disabled}
          size="sm"
          aria-label="Use a reference recording instead of a voice"
          data-testid="clone-toggle"
        />
      }
    >
      <div className="grid gap-3">
        <p className="text-[13px] leading-5 text-muted-foreground">
          Give a few seconds (up to 30) of clear speech from one person. It is sent with each request and replaces the
          chosen voice. Only clone voices you have permission to use: your own, or one whose speaker agreed.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => fileInput.current?.click()}
            disabled={disabled || !enabled || recording}
          >
            <Upload /> Upload a recording
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
            className="sr-only"
            tabIndex={-1}
            aria-label="Reference recording"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void onFile(f)
              e.target.value = ''
            }}
          />
          {recording ? (
            <Button size="sm" variant="destructive" onClick={() => void finishRecording()} data-testid="clone-stop">
              <Square className="fill-current" /> Stop · <span className="num">{fmtSeconds(elapsed)}</span>
            </Button>
          ) : (
            <Button
              size="sm"
              variant="outline"
              onClick={record}
              disabled={disabled || !enabled}
              data-testid="clone-record"
            >
              <Mic /> Record
            </Button>
          )}
          {reference && (
            <>
              <span className="min-w-0 truncate text-xs text-muted-foreground" data-testid="clone-reference">
                {reference.label}
                {reference.seconds != null && <span className="num"> · {fmtSeconds(reference.seconds)}</span>}
              </span>
              <Button size="icon-sm" variant="ghost" onClick={play} aria-label="Play the reference recording">
                <Volume2 />
              </Button>
              {onSave && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7"
                  onClick={() => onSave(reference)}
                  disabled={disabled || recording}
                  data-testid="clone-save"
                >
                  <BookmarkPlus /> Save as a voice
                </Button>
              )}
              <Button
                size="icon-sm"
                variant="ghost"
                onClick={() => onReferenceChange(null)}
                aria-label="Remove the reference recording"
              >
                <Trash2 />
              </Button>
            </>
          )}
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-text`} className="text-xs">
            What is said in the recording (optional; some models use it)
          </Label>
          <Textarea
            id={`${id}-text`}
            value={reference?.text ?? ''}
            onChange={(e) => reference && onReferenceChange({ ...reference, text: e.target.value })}
            disabled={disabled || !enabled || !reference}
            placeholder={reference ? 'Type the words of the recording' : 'Add a recording first'}
            className="min-h-16 text-[14px]"
          />
        </div>
        {enabled && <p className="text-xs text-muted-foreground">Cloned speech is sent over chunked HTTP.</p>}
      </div>
    </Panel>
  )
}
