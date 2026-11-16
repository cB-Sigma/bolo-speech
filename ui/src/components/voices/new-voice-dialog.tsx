import { Mic, Square, TriangleAlert, Upload } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { describeError } from '@/hooks/use-api'
import type { SavedVoice } from '@/lib/api/types'
import { openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import { encodeWav } from '@/lib/audio/wav'
import { fmtSeconds } from '@/lib/format'
import { CONSENT_LABEL, scoreText, useVoiceMutations } from './use-voices'

/** The server keeps 2 to 60 s of speech; most models clone best from 5 to 30 s. */
export const MAX_RECORDING_S = 60

export interface VoiceSource {
  blob: Blob
  name: string
  seconds: number | null
  transcript?: string
}

/** A `data:` URL (the clone panel's reference) as a Blob. */
export function dataUrlToBlob(dataUrl: string): Blob {
  const [head, data = ''] = dataUrl.split(',', 2)
  const type = /data:([^;,]+)/.exec(head ?? '')?.[1] ?? 'application/octet-stream'
  const bin = atob(data)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type })
}

/**
 * Saves a voice to the server's library: a recording (uploaded, recorded here, or handed over by the clone panel),
 * a name, and an explicit consent statement, which the server requires. With a model that clones voices, a test
 * sentence is spoken and scored against the recording.
 */
export function NewVoiceDialog({
  open,
  onOpenChange,
  initial,
  scoreModel,
  onCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initial?: VoiceSource | null
  /** A model that clones voices: speak a test sentence with it and show the speaker similarity. */
  scoreModel?: string | null
  onCreated?: (voice: SavedVoice) => void
}) {
  const id = useId()
  const { create } = useVoiceMutations()
  const [name, setName] = useState('')
  const [language, setLanguage] = useState('')
  // Mounted afresh for each opening (the library renders it only while open), so the fields start from `initial`.
  const [transcript, setTranscript] = useState(initial?.transcript ?? '')
  const [source, setSource] = useState<VoiceSource | null>(initial ?? null)
  const [basis, setBasis] = useState<'' | 'self' | 'permission'>('')
  const [confirmed, setConfirmed] = useState(false)
  const [score, setScore] = useState(true)
  const [created, setCreated] = useState<SavedVoice | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
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

  const stop = async () => {
    wanted.current = false
    clearInterval(timer.current)
    const m = mic.current
    mic.current = null
    setRecording(false)
    await m?.stop()
    const audio = concatFloat(blocks.current)
    blocks.current = []
    const seconds = audio.length / STT_RATE
    if (seconds < 2) {
      toast.error('The recording is too short', { description: 'Record at least a few seconds of clear speech.' })
      return
    }
    setSource({
      blob: new Blob([encodeWav(audio, STT_RATE) as BlobPart], { type: 'audio/wav' }),
      name: 'recording.wav',
      seconds,
    })
  }

  const record = () => {
    // The permission prompt must come straight from the tap (iOS Safari).
    blocks.current = []
    wanted.current = true
    const opening = openMicrophone((block) => blocks.current.push(block))
    setRecording(true)
    setElapsed(0)
    const started = performance.now()
    timer.current = setInterval(() => {
      const s = (performance.now() - started) / 1000
      setElapsed(s)
      if (s >= MAX_RECORDING_S) void stop()
    }, 200)
    opening.then(
      (m) => {
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

  const ready = name.trim() !== '' && source != null && basis !== '' && confirmed && !recording
  const submit = () => {
    if (!ready || !source) return
    create.mutate(
      {
        name: name.trim(),
        files: [{ blob: source.blob, name: source.name }],
        consent: basis,
        transcript: transcript.trim() || undefined,
        language: language.trim() || undefined,
        scoreModel: score && scoreModel ? scoreModel : undefined,
      },
      {
        onSuccess: (voice) => {
          setCreated(voice)
          onCreated?.(voice)
          toast.success(`Saved the voice ${voice.name}`)
        },
        onError: (err) => toast.error('Could not save the voice', { description: describeError(err) }),
      },
    )
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !recording && onOpenChange(o)}>
      <DialogContent className="sm:max-w-lg" data-testid="new-voice-dialog">
        <DialogHeader>
          <DialogTitle>{created ? `Saved: ${created.name}` : 'Save a voice'}</DialogTitle>
          <DialogDescription>
            {created
              ? 'The cleaned recording is stored on the server. Name it as the voice of any model that clones voices.'
              : 'A recording of one person speaking clearly, 5 to 30 s works best. The server trims silence and evens out the loudness.'}
          </DialogDescription>
        </DialogHeader>
        {created ? (
          <div className="grid gap-2 text-[13px] leading-5" data-testid="new-voice-result">
            <p>
              <span className="num">{created.id}</span> · {fmtSeconds(created.reference.duration_s)} of reference
              {created.transcript_source === 'auto' ? ' · transcript from speech recognition' : ''}
            </p>
            {created.warnings.map((w) => (
              <p key={w} className="flex items-start gap-1.5 text-warning-text">
                <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden /> {w}
              </p>
            ))}
            {created.score && (
              <p data-testid="new-voice-score">
                Speaker similarity with <span className="num">{created.score.model}</span>:{' '}
                <span className="num">{scoreText(created.score)}</span>
              </p>
            )}
          </div>
        ) : (
          <div className="grid gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-name`} className="text-xs">
                Name
              </Label>
              <Input
                id={`${id}-name`}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Priya"
                maxLength={100}
                data-testid="new-voice-name"
              />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => fileInput.current?.click()} disabled={recording}>
                <Upload /> Upload
              </Button>
              <input
                ref={fileInput}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
                className="sr-only"
                tabIndex={-1}
                aria-label="Voice recording"
                data-testid="new-voice-file"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) setSource({ blob: f, name: f.name, seconds: null })
                  e.target.value = ''
                }}
              />
              {recording ? (
                <Button size="sm" variant="destructive" onClick={() => void stop()}>
                  <Square className="fill-current" /> Stop · <span className="num">{fmtSeconds(elapsed)}</span>
                </Button>
              ) : (
                <Button size="sm" variant="outline" onClick={record}>
                  <Mic /> Record
                </Button>
              )}
              <span className="min-w-0 truncate text-xs text-muted-foreground" data-testid="new-voice-source">
                {source
                  ? `${source.name}${source.seconds != null ? ` · ${fmtSeconds(source.seconds)}` : ''}`
                  : 'No recording yet'}
              </span>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-text`} className="text-xs">
                What is said in it (optional: the server transcribes it when a recogniser is installed)
              </Label>
              <Textarea
                id={`${id}-text`}
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
                className="min-h-14 text-[14px]"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-lang`} className="text-xs">
                Language (optional, a code such as en or de)
              </Label>
              <Input
                id={`${id}-lang`}
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-32"
                maxLength={16}
              />
            </div>
            <fieldset className="grid gap-2 rounded-lg border border-border p-3" data-testid="new-voice-consent">
              <legend className="px-1 text-xs font-medium">Consent</legend>
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={basis}
                onValueChange={(v) => setBasis((v || '') as '' | 'self' | 'permission')}
                aria-label="Whose voice this is"
                className="flex-wrap justify-start"
              >
                <ToggleGroupItem value="self" className="px-3 text-xs">
                  {CONSENT_LABEL.self}
                </ToggleGroupItem>
                <ToggleGroupItem value="permission" className="px-3 text-xs">
                  {CONSENT_LABEL.permission}
                </ToggleGroupItem>
              </ToggleGroup>
              <label className="flex items-start gap-2 text-[13px] leading-5">
                <Checkbox
                  checked={confirmed}
                  onCheckedChange={(c) => setConfirmed(c === true)}
                  disabled={basis === ''}
                  aria-label="I confirm the consent statement"
                  data-testid="new-voice-confirm"
                  className="mt-0.5"
                />
                <span>
                  I confirm the statement above. The server records it with the voice, with the time and who made it.
                </span>
              </label>
            </fieldset>
            {scoreModel && (
              <label className="flex items-center gap-2 text-[13px]">
                <Checkbox checked={score} onCheckedChange={(c) => setScore(c === true)} aria-label="Score the clone" />
                <span>
                  Speak a test sentence with <span className="num">{scoreModel}</span> and score the likeness
                </span>
              </label>
            )}
          </div>
        )}
        <DialogFooter className="gap-2">
          {created ? (
            <Button onClick={() => onOpenChange(false)}>Done</Button>
          ) : (
            <>
              <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={recording}>
                Cancel
              </Button>
              <Button onClick={submit} disabled={!ready || create.isPending} data-testid="new-voice-save">
                {create.isPending ? (score && scoreModel ? 'Saving and scoring…' : 'Saving…') : 'Save voice'}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
