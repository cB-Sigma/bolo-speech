import { Download, Mic, Play, Square, Upload } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { Panel } from '@/components/panel'
import { VoiceClonePanel, type CloneReference } from '@/components/tts/voice-clone'
import { dataUrlToBlob } from '@/components/voices/new-voice-dialog'
import { useVoices } from '@/components/voices/use-voices'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { describeError, pickDefault, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import { encodeWav } from '@/lib/audio/wav'
import { download } from '@/lib/bench'
import { REFERENCE_CLIPS } from '@/lib/dialogue'
import { fmtSeconds } from '@/lib/format'

/** Models with voices of their own (RVC) convert to one of those instead of a recording. */
function ownVoices(model: { voices: { id: string; name: string }[] } | null): { id: string; name: string }[] {
  return model?.voices ?? []
}

interface Source {
  blob: Blob
  label: string
  url: string
}

interface Result {
  url: string
  bytes: Uint8Array
  model: string
  target: string
}

/** Voice conversion: a recording of speech in, the same words out in another voice. */
export default function ConvertPage() {
  const { models, isLoading } = useModels('voice-conversion')
  const [modelId, setModelId] = useState<string | null>(null)
  const model = useMemo(() => pickDefault(models, modelId), [models, modelId])
  const library = useVoices()
  const [source, setSource] = useState<Source | null>(null)
  const [target, setTarget] = useState<string>('en-short')
  const [useUpload, setUseUpload] = useState(false)
  const [reference, setReference] = useState<CloneReference | null>(null)
  const [busy, setBusy] = useState(false)
  const [watermark, setWatermark] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const mic = useRef<AudioSource | null>(null)
  const blocks = useRef<Float32Array[]>([])
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined)
  const fileInput = useRef<HTMLInputElement>(null)
  const own = ownVoices(model)
  const needsRecording = own.length === 0
  const saved = useMemo(() => library.data?.voices ?? [], [library.data])
  // a model with its own voices takes one of them; the others a recording (a clip or a saved voice)
  const valid = own.length
    ? own.some((v) => v.id === target)
    : REFERENCE_CLIPS.includes(target) || saved.some((v) => v.id === target)
  const chosen = valid ? target : own.length ? (own[0]?.id ?? '') : 'en-short'

  useEffect(() => () => void (source && URL.revokeObjectURL(source.url)), [source])
  useEffect(() => () => void (result && URL.revokeObjectURL(result.url)), [result])
  useEffect(
    () => () => {
      clearInterval(timer.current)
      void mic.current?.stop()
    },
    [],
  )

  const setBlob = (blob: Blob, label: string) => setSource({ blob, label, url: URL.createObjectURL(blob) })

  const record = () => {
    blocks.current = []
    setRecording(true)
    setElapsed(0)
    const started = performance.now()
    timer.current = setInterval(() => setElapsed((performance.now() - started) / 1000), 200)
    openMicrophone((block) => blocks.current.push(block)).then(
      (m) => {
        mic.current = m
      },
      (err: unknown) => {
        clearInterval(timer.current)
        setRecording(false)
        toast.error('Could not open the microphone', { description: err instanceof Error ? err.message : undefined })
      },
    )
  }

  const stop = async () => {
    clearInterval(timer.current)
    setRecording(false)
    const m = mic.current
    mic.current = null
    await m?.stop()
    const audio = concatFloat(blocks.current)
    if (audio.length < STT_RATE * 0.5) {
      toast.error('The recording is too short')
      return
    }
    setBlob(new Blob([encodeWav(audio, STT_RATE) as BlobPart], { type: 'audio/wav' }), 'Recorded here')
  }

  const convert = async () => {
    if (!model || !source) return
    setBusy(true)
    try {
      const upload = needsRecording && useUpload && reference ? dataUrlToBlob(reference.dataUrl) : undefined
      const res = await api.convertVoice(source.blob, {
        model: model.id,
        voice: upload ? undefined : chosen,
        reference: upload,
        watermark,
      })
      const blob = await res.blob()
      setResult({
        url: URL.createObjectURL(blob),
        bytes: new Uint8Array(await blob.arrayBuffer()),
        model: res.headers.get('x-bolo-model') ?? model.id,
        target: res.headers.get('x-bolo-target') ?? chosen,
      })
    } catch (err) {
      toast.error('The conversion failed', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  const toolbar = (
    <ModelPicker
      models={models}
      value={model?.id ?? null}
      onChange={setModelId}
      loading={isLoading}
      disabled={busy}
      className="w-64 max-sm:min-w-44 max-sm:flex-1"
    />
  )

  return (
    <Page title="Voice conversion" toolbar={toolbar}>
      {model && !model.installed && (
        <Alert className="mb-4">
          <AlertTitle>
            <span className="num">{model.id}</span> is not installed
          </AlertTitle>
          <AlertDescription>Pull it from the Models screen first.</AlertDescription>
        </Alert>
      )}
      {model && model.commercial_use === false && (
        <Alert className="mb-4" data-testid="convert-licence">
          <AlertTitle>Licence: {model.license}</AlertTitle>
          <AlertDescription>Not cleared for commercial use; see the model's licence.</AlertDescription>
        </Alert>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Speech to convert" description="Its words, rhythm and intonation are kept">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => fileInput.current?.click()} disabled={recording}>
                <Upload /> Upload a recording
              </Button>
              <input
                ref={fileInput}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
                className="sr-only"
                tabIndex={-1}
                aria-label="Speech to convert"
                data-testid="convert-source"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) setBlob(f, f.name)
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
            </div>
            {source && (
              <div className="mt-3 grid gap-1.5">
                <span className="text-xs text-muted-foreground">{source.label}</span>
                <audio controls src={source.url} className="w-full" />
              </div>
            )}
          </Panel>
          <Panel
            title="Target voice"
            description={
              needsRecording
                ? 'This model imitates a voice from a recording: a saved voice, a reference clip or one you upload'
                : 'This model converts to one of its own voices'
            }
          >
            <Select value={chosen} onValueChange={setTarget} disabled={busy || (needsRecording && useUpload)}>
              <SelectTrigger className="h-8 w-full" aria-label="Target voice" data-testid="convert-target">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start" className="max-h-72">
                {needsRecording ? (
                  <>
                    <SelectGroup>
                      <SelectLabel className="text-xs">Reference clips</SelectLabel>
                      {REFERENCE_CLIPS.map((c) => (
                        <SelectItem key={c} value={c}>
                          {c}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                    {saved.length > 0 && (
                      <SelectGroup>
                        <SelectLabel className="text-xs">Saved voices</SelectLabel>
                        {saved.map((v) => (
                          <SelectItem key={v.id} value={v.id}>
                            {v.name}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    )}
                  </>
                ) : (
                  own.map((v) => (
                    <SelectItem key={v.id} value={v.id}>
                      {v.name || v.id}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </Panel>
          {needsRecording && (
            <VoiceClonePanel
              enabled={useUpload}
              onEnabledChange={setUseUpload}
              reference={reference}
              onReferenceChange={setReference}
              disabled={busy}
            />
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Converted">
            <Button
              onClick={() => void convert()}
              disabled={
                busy || !source || !model?.installed || recording || (needsRecording && useUpload && !reference)
              }
              className="h-9 min-w-28"
              data-testid="convert-run"
            >
              <Play className="fill-current" /> {busy ? 'Converting…' : 'Convert'}
            </Button>
            <label
              className="ml-2 inline-flex h-8 items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
              title="An inaudible mark that says the speech was converted by a machine (AudioSeal)"
            >
              Watermark
              <Switch
                checked={watermark}
                onCheckedChange={setWatermark}
                size="sm"
                disabled={busy}
                aria-label="Add an inaudible watermark"
                data-testid="convert-watermark"
              />
            </label>
            {result && (
              <div className="mt-4 grid gap-3" data-testid="convert-result">
                <audio controls src={result.url} className="w-full" />
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  <dt className="text-muted-foreground">Model</dt>
                  <dd className="text-right num break-all">{result.model}</dd>
                  <dt className="text-muted-foreground">Target</dt>
                  <dd className="text-right num break-all">{result.target}</dd>
                </dl>
                <div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => download(`bolo-converted-${result.model}.wav`, result.bytes, 'audio/wav')}
                  >
                    <Download /> Save WAV
                  </Button>
                </div>
              </div>
            )}
            <p className="mt-4 text-xs leading-5 text-muted-foreground">
              Only convert to voices you have permission to use: your own, or one whose speaker agreed.
            </p>
          </Panel>
        </div>
      </div>
    </Page>
  )
}
