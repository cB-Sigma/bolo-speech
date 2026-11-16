import {
  AudioLines,
  Download,
  FileUp,
  Loader2,
  Mic,
  Play,
  Save,
  ShieldCheck,
  Square,
  Trash2,
  TriangleAlert,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { DownloadsPanel, InterruptedLoadNotice } from '@/components/downloads-panel'
import { Page } from '@/components/layout/app-shell'
import { RiskDialog, useDevice, VerdictBadge, VerdictLine } from '@/components/model-fit'
import { Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useStore } from '@/hooks/use-store'
import { GaplessPlayer } from '@/lib/audio/player'
import { resample } from '@/lib/audio/resample'
import { encodeWav } from '@/lib/audio/wav'
import {
  decodeToCloneRate,
  exampleUrl,
  fromInt16,
  loadExamples,
  MAX_SECONDS,
  MIN_SECONDS,
  prepareReference,
  startRecorder,
  toInt16,
  type CloneExample,
  type Recorder,
} from '@/lib/browser-clone/audio'
import { sharedCloner, sharedScorer, type GenerateResult, type Placement } from '@/lib/browser-clone/engine'
import {
  CLONE_GROUPS,
  CLONE_MODELS,
  cloneDownloadId,
  LANGUAGE_NAMES,
  NOT_OFFERED,
  type CloneModel,
} from '@/lib/browser-clone/models'
import { CLONE_RATE } from '@/lib/browser-clone/protocol'
import { savedVoices } from '@/lib/browser-clone/saved-voices'
import {
  cosine,
  DIFFERENT_SPEAKER,
  forEmbedding,
  SAME_SPEAKER,
  SIMILARITY_MODEL,
  VERDICT_WORDS,
  verdictFor as similarityVerdict,
} from '@/lib/browser-clone/similarity'
import { wordErrorRate } from '@/lib/browser-clone/text'
import { recogniserFor } from '@/lib/browser-diarize/examples'
import { isPhoneLike } from '@/lib/browser-runtime'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { browserModel } from '@/lib/browser-stt/models'
import { peakFor, verdictFor, type DeviceInfo } from '@/lib/device'
import { isDownloaded } from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

interface Reference {
  /** Identifies the audio for the model (a saved voice's id, an example's id, or a fresh one). */
  key: string
  name: string
  audio: Float32Array
  seconds: number
  notes: string[]
  source: 'mic' | 'upload' | 'example' | 'saved'
  language: string
  savedId?: string
}

interface Output {
  result: GenerateResult
  text: string
  model: CloneModel
  url: string
  similarity: { score: number; ms: number } | 'busy' | { error: string } | null
  roundTrip: { text: string; wer: number; model: string } | 'busy' | null
}

const DEFAULT_TEXT: Record<string, string> = {
  en: 'Thanks for calling. Your appointment is confirmed for Tuesday at half past ten, and we will send a reminder the day before.',
  hi: 'नमस्ते, आपका अपॉइंटमेंट मंगलवार सुबह साढ़े दस बजे के लिए पक्का हो गया है।',
  de: 'Vielen Dank für Ihren Anruf. Ihr Termin am Dienstag um halb elf ist bestätigt.',
  es: 'Gracias por llamar. Su cita del martes a las diez y media está confirmada.',
  fr: 'Merci de votre appel. Votre rendez-vous de mardi à dix heures et demie est confirmé.',
  it: 'Grazie per aver chiamato. Il suo appuntamento di martedì alle dieci e mezza è confermato.',
  pt: 'Obrigado por ligar. A sua consulta de terça-feira às dez e meia está confirmada.',
}

const RECOGNISED = ['en', 'hi', 'de', 'es', 'fr', 'it', 'pt']

let freshKey = 0
const newKey = (prefix: string) => `${prefix}-${Date.now()}-${++freshKey}`

function needsFor(m: CloneModel) {
  const id = cloneDownloadId(m)
  return {
    bytes: m.bytes,
    peakBytes: peakFor(id, m.peakBytes),
    prefersWebGpu: m.prefersWebGpu,
    downloaded: isDownloaded(id),
  }
}

function ModelOption({ m, d }: { m: CloneModel; d: DeviceInfo }) {
  const v = verdictFor(needsFor(m), d)
  return (
    <span className="grid min-w-0">
      <span className="flex min-w-0 flex-wrap items-center gap-1.5">
        <span className="truncate">{m.label}</span>
        <VerdictBadge verdict={v} />
      </span>
      <span className="text-[11px] whitespace-normal text-muted-foreground">
        {fmtBytes(m.bytes)} · {m.license} ·{' '}
        {m.languages.length > 3
          ? `${m.languages.length} languages`
          : m.languages.map((l) => LANGUAGE_NAMES[l] ?? l).join(', ')}
      </span>
    </span>
  )
}

function SimilarityView({ s }: { s: Output['similarity'] }) {
  if (s == null) return null
  if (s === 'busy')
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground" data-testid="similarity">
        <Loader2 className="size-3 animate-spin" aria-hidden /> Comparing the voices…
      </p>
    )
  if ('error' in s)
    return (
      <p className="text-xs text-destructive" data-testid="similarity">
        Could not score the similarity: {s.error}
      </p>
    )
  const v = similarityVerdict(s.score)
  return (
    <p className="text-xs" data-testid="similarity" data-score={s.score.toFixed(3)} data-verdict={v}>
      <span className="font-medium">Speaker similarity {s.score.toFixed(2)}</span>:{' '}
      <span
        className={cn(
          v === 'same' ? 'text-success-text' : v === 'different' ? 'text-destructive' : 'text-warning-text',
          'font-medium',
        )}
      >
        {VERDICT_WORDS[v]}
      </span>
      <span className="text-muted-foreground">
        {' '}
        (cosine of {SIMILARITY_MODEL.label} embeddings; {SAME_SPEAKER} or more: likely the same, under{' '}
        {DIFFERENT_SPEAKER}: likely different)
      </span>
    </p>
  )
}

export default function CloneBrowserPage() {
  const cloner = sharedCloner()
  const s = useStore(cloner.store)
  const saved = savedVoices()
  const savedState = useStore(saved.store)
  const d = useDevice()
  const [examples, setExamples] = useState<CloneExample[]>([])
  const [ref, setRef] = useState<Reference | null>(null)
  const [consent, setConsent] = useState(false)
  const [recorder, setRecorder] = useState<{ rec: Recorder; started: number } | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [voiceName, setVoiceName] = useState('')
  const [language, setLanguage] = useState('en')
  const [text, setText] = useState(DEFAULT_TEXT.en as string)
  const [output, setOutput] = useState<Output | null>(null)
  const [scoreAuto, setScoreAuto] = useState(() => !isPhoneLike())
  const [risk, setRisk] = useState(false)
  const playerRef = useRef<GaplessPlayer | null>(null)
  const model = s.model
  const verdict = verdictFor(needsFor(model), d)

  useEffect(() => {
    void saved.load()
    loadExamples()
      .then(setExamples)
      .catch(() => setExamples([]))
  }, [saved])

  useEffect(() => {
    if (!recorder) return
    const t = setInterval(() => {
      const sec = (performance.now() - recorder.started) / 1000
      setElapsed(sec)
      if (sec >= MAX_SECONDS) void stopRecording()
    }, 200)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recorder])

  useEffect(() => () => void playerRef.current?.close(), [])

  // The model speaks the language picked, or its own when it has only one.
  const textLanguage = model.languages.includes(language) ? language : (model.languages[0] as string)

  const transcribe = async (audio: Float32Array, lang: string) => {
    setTranscribing(true)
    try {
      const engine = sharedEngine()
      engine.setModel(recogniserFor(lang, engine.store.get().model.id))
      const r = await engine.transcribe(resample(audio, CLONE_RATE, 16_000), { language: lang })
      return r.text.trim()
    } finally {
      setTranscribing(false)
    }
  }

  const chooseReference = (r: Reference, knownTranscript?: string) => {
    setRef(r)
    setOutput(null)
    setVoiceName(r.name)
    if (knownTranscript != null) setTranscript(knownTranscript)
    else {
      setTranscript('')
      transcribe(r.audio, r.language)
        .then(setTranscript)
        .catch((err: unknown) =>
          toast.error('Could not transcribe the recording', {
            description: err instanceof Error ? err.message : String(err),
          }),
        )
    }
  }

  const fromAudio = (raw: Float32Array, source: Reference['source'], name: string) => {
    const p = prepareReference(raw)
    chooseReference({ key: newKey(source), name, audio: p.audio, seconds: p.seconds, notes: p.notes, source, language })
  }

  const pickExample = async (ex: CloneExample) => {
    try {
      const res = await fetch(exampleUrl(ex.file))
      const audio = await decodeToCloneRate(await res.arrayBuffer())
      const p = prepareReference(audio)
      chooseReference(
        {
          key: `example-${ex.id}`,
          name: ex.title,
          audio: p.audio,
          seconds: p.seconds,
          notes: p.notes,
          source: 'example',
          language: ex.language,
        },
        ex.transcript,
      )
      if (model.languages.includes(ex.language)) {
        setLanguage(ex.language)
        setText(DEFAULT_TEXT[ex.language] ?? text)
      }
    } catch (err) {
      toast.error('Could not load the example', { description: err instanceof Error ? err.message : String(err) })
    }
  }

  const upload = async (file: File) => {
    try {
      fromAudio(await decodeToCloneRate(await file.arrayBuffer()), 'upload', file.name.replace(/\.[^.]+$/, ''))
    } catch {
      toast.error('This browser cannot read that file', { description: 'Try WAV, MP3, M4A, OGG or WebM.' })
    }
  }

  const startRecording = async () => {
    try {
      setRecorder({ rec: await startRecorder(), started: performance.now() })
      setElapsed(0)
    } catch (err) {
      toast.error('No microphone', { description: err instanceof Error ? err.message : String(err) })
    }
  }

  async function stopRecording() {
    const r = recorder
    if (!r) return
    setRecorder(null)
    try {
      fromAudio(await r.rec.stop(), 'mic', 'My recording')
    } catch (err) {
      toast.error('Could not read the recording', { description: err instanceof Error ? err.message : String(err) })
    }
  }

  const saveVoice = async () => {
    if (!ref) return
    const id = ref.savedId ?? `voice-${Date.now()}`
    await saved.save({
      id,
      name: voiceName.trim() || ref.name,
      createdAt: Date.now(),
      sampleRate: CLONE_RATE,
      pcm: toInt16(ref.audio),
      transcript,
      language: ref.language,
      source: ref.source === 'saved' ? 'upload' : ref.source,
    })
    setRef({ ...ref, savedId: id, key: id, source: 'saved', name: voiceName.trim() || ref.name })
    toast.success(saved.store.get().persistent ? 'Saved in this browser' : 'Kept for this visit only')
  }

  const pickSaved = (id: string) => {
    const v = saved.get(id)
    if (!v) return
    const audio = fromInt16(v.pcm)
    chooseReference(
      {
        key: v.id,
        name: v.name,
        audio,
        seconds: audio.length / v.sampleRate,
        notes: [],
        source: 'saved',
        language: v.language,
        savedId: v.id,
      },
      v.transcript,
    )
  }

  const score = async (out: Output, reference: Reference) => {
    setOutput((o) => (o && o.url === out.url ? { ...o, similarity: 'busy' } : o))
    try {
      const scorer = sharedScorer()
      const started = performance.now()
      const a = await scorer.embed(forEmbedding(reference.audio))
      const b = await scorer.embed(forEmbedding(out.result.audio))
      const result = { score: cosine(a, b), ms: performance.now() - started }
      setOutput((o) => (o && o.url === out.url ? { ...o, similarity: result } : o))
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err)
      setOutput((o) => (o && o.url === out.url ? { ...o, similarity: { error } } : o))
    }
  }

  const roundTrip = async (out: Output) => {
    setOutput((o) => (o && o.url === out.url ? { ...o, roundTrip: 'busy' } : o))
    try {
      const heard = await transcribe(out.result.audio, textLanguage)
      const engineModel = browserModel(sharedEngine().store.get().model.id).label
      setOutput((o) =>
        o && o.url === out.url
          ? { ...o, roundTrip: { text: heard, wer: wordErrorRate(out.text, heard), model: engineModel } }
          : o,
      )
    } catch (err) {
      toast.error('Could not transcribe the result', { description: err instanceof Error ? err.message : String(err) })
      setOutput((o) => (o && o.url === out.url ? { ...o, roundTrip: null } : o))
    }
  }

  const generate = async (confirmed = false) => {
    if (!ref || !text.trim()) return
    if (!confirmed && s.status !== 'ready' && verdict.level === 'unlikely') {
      setRisk(true)
      return
    }
    void playerRef.current?.close()
    const player = new GaplessPlayer()
    playerRef.current = player
    if (output) URL.revokeObjectURL(output.url)
    setOutput(null)
    try {
      const result = await cloner.generate(
        ref.key,
        ref.audio,
        text.trim(),
        { language: textLanguage, temperature: 0.7, lsdSteps: 1, exaggeration: 0.5 },
        (a) => player.enqueue(a, CLONE_RATE),
      )
      player.end()
      const url = URL.createObjectURL(
        new Blob([encodeWav(result.audio, CLONE_RATE) as BlobPart], { type: 'audio/wav' }),
      )
      const out: Output = { result, text: text.trim(), model, url, similarity: null, roundTrip: null }
      setOutput(out)
      if (scoreAuto && result.audio.length > CLONE_RATE * 0.5) void score(out, ref)
    } catch (err) {
      player.end()
      const message = err instanceof Error ? err.message : String(err)
      if (message !== 'Unloaded') toast.error('Could not generate', { description: message })
    }
  }

  const grouped = useMemo(() => CLONE_GROUPS.map((g) => ({ g, models: CLONE_MODELS.filter((m) => m.group === g) })), [])
  const busy = s.busy || s.status === 'loading'
  const canUseOwn = consent

  return (
    <Page title="Clone a voice">
      <InterruptedLoadNotice />
      <Alert className="mb-4" data-testid="clone-privacy">
        <ShieldCheck />
        <AlertTitle>Everything stays in this browser</AlertTitle>
        <AlertDescription>
          The models run in this page. Your recordings, the voices you save and what you generate are never uploaded:
          the only downloads are the model files, from Hugging Face. Only copy a voice that is yours or that you have
          permission to use.
        </AlertDescription>
      </Alert>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="grid min-w-0 content-start gap-4">
          <Panel
            title="1. The voice to copy"
            description={`${MIN_SECONDS} to ${MAX_SECONDS} seconds of one person speaking`}
            data-testid="clone-reference"
          >
            <div className="grid gap-4">
              <section className="grid gap-2" aria-labelledby="clone-examples-h">
                <h3 id="clone-examples-h" className="text-xs font-medium">
                  Example voices <span className="font-normal text-muted-foreground">(synthetic speech)</span>
                </h3>
                <ul className="grid gap-2 sm:grid-cols-3" data-testid="clone-examples">
                  {examples.map((ex) => (
                    <li key={ex.id} className="grid min-w-0 gap-1.5 rounded-lg border border-border p-2.5">
                      <span className="text-xs font-medium">{ex.title}</span>
                      <audio
                        controls
                        preload="none"
                        src={exampleUrl(ex.file)}
                        className="h-8 w-full"
                        aria-label={`Play the example: ${ex.title}`}
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void pickExample(ex)}
                        data-testid={`clone-example-${ex.id}`}
                      >
                        <AudioLines /> Use this voice
                      </Button>
                      <span className="text-[11px] leading-4 text-muted-foreground">
                        Synthetic: {ex.credit}. {ex.seconds.toFixed(0)} s.
                      </span>
                    </li>
                  ))}
                </ul>
              </section>

              <section className="grid gap-2" aria-labelledby="clone-own-h">
                <h3 id="clone-own-h" className="text-xs font-medium">
                  Your own recording
                </h3>
                <label className="flex items-start gap-2 text-xs" htmlFor="clone-consent">
                  <Checkbox
                    id="clone-consent"
                    checked={consent}
                    onCheckedChange={(v) => setConsent(v === true)}
                    data-testid="clone-consent"
                    className="mt-0.5"
                  />
                  <span>
                    This is my voice, or I have the speaker’s permission to copy it. I will not use the copy to deceive
                    anyone.
                  </span>
                </label>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    variant={recorder ? 'destructive' : 'outline'}
                    onClick={() => void (recorder ? stopRecording() : startRecording())}
                    disabled={!canUseOwn && !recorder}
                    data-testid="clone-record"
                  >
                    {recorder ? <Square /> : <Mic />}{' '}
                    {recorder ? `Stop (${elapsed.toFixed(0)} s of ${MAX_SECONDS})` : 'Record'}
                  </Button>
                  <Button size="sm" variant="outline" asChild disabled={!canUseOwn}>
                    <label className={cn('cursor-pointer', !canUseOwn && 'pointer-events-none opacity-50')}>
                      <FileUp /> Upload audio
                      <input
                        type="file"
                        accept="audio/*,video/*"
                        className="sr-only"
                        disabled={!canUseOwn}
                        data-testid="clone-upload"
                        onChange={(e) => {
                          const f = e.target.files?.[0]
                          if (f) void upload(f)
                          e.target.value = ''
                        }}
                      />
                    </label>
                  </Button>
                  <label className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Spoken in</span>
                    <Select value={language} onValueChange={setLanguage}>
                      <SelectTrigger className="h-7 w-32" aria-label="Language of the recording and the text">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {RECOGNISED.map((l) => (
                          <SelectItem key={l} value={l}>
                            {LANGUAGE_NAMES[l]}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                </div>
                {!canUseOwn && (
                  <p className="text-[11px] text-muted-foreground">Tick the box above to record or upload.</p>
                )}
              </section>

              {savedState.voices.length > 0 && (
                <section className="grid gap-2" aria-labelledby="clone-saved-h">
                  <h3 id="clone-saved-h" className="text-xs font-medium">
                    Saved in this browser
                  </h3>
                  <ul className="grid gap-1.5" data-testid="clone-saved">
                    {savedState.voices.map((v) => (
                      <li key={v.id} className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
                        <span className="min-w-0 flex-1 truncate font-medium">{v.name}</span>
                        <span className="text-muted-foreground">{LANGUAGE_NAMES[v.language] ?? v.language}</span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => pickSaved(v.id)}
                          data-testid="clone-use-saved"
                        >
                          Use
                        </Button>
                        <Button
                          size="icon-sm"
                          variant="ghost"
                          aria-label={`Delete the saved voice ${v.name}`}
                          onClick={() => void saved.remove(v.id)}
                          data-testid="clone-delete-saved"
                        >
                          <Trash2 />
                        </Button>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {!savedState.persistent && (
                <p className="text-[11px] text-warning-text" data-testid="clone-not-persistent">
                  This browser does not let the page keep voices (a private window, or storage is blocked): saved voices
                  last until the page is closed.
                </p>
              )}

              {ref && (
                <section className="grid gap-2 rounded-lg border border-border p-3" data-testid="clone-chosen">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <h3 className="text-sm font-medium">{ref.name}</h3>
                    <span className="text-xs text-muted-foreground" data-testid="clone-ref-seconds">
                      {ref.seconds.toFixed(1)} s · {LANGUAGE_NAMES[ref.language] ?? ref.language}
                    </span>
                  </div>
                  <RefPlayer audio={ref.audio} />
                  {ref.notes.map((n) => (
                    <p key={n} className="flex items-start gap-1.5 text-xs text-warning-text">
                      <TriangleAlert className="mt-0.5 size-3 shrink-0" aria-hidden /> {n}
                    </p>
                  ))}
                  {model.maxReferenceSeconds != null && ref.seconds > model.maxReferenceSeconds && (
                    <p className="text-xs text-muted-foreground" data-testid="clone-ref-cut">
                      {model.label} uses the first {model.maxReferenceSeconds} s of it.
                    </p>
                  )}
                  <label className="grid gap-1 text-xs">
                    <span className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">What is said in the recording</span>
                      <span className="text-muted-foreground">
                        {transcribing ? 'listening…' : 'filled in by the recogniser in this page; correct it if needed'}
                      </span>
                    </span>
                    <Textarea
                      value={transcript}
                      onChange={(e) => setTranscript(e.target.value)}
                      rows={3}
                      placeholder={transcribing ? 'Transcribing…' : 'The words of the recording'}
                      data-testid="clone-transcript"
                    />
                    <span className="text-[11px] text-muted-foreground">
                      Kept with a saved voice. None of the models offered here needs it: they take the voice from the
                      sound alone.
                    </span>
                  </label>
                  <div className="flex flex-wrap items-end gap-2">
                    <label className="grid min-w-0 flex-1 gap-1 text-xs">
                      <span className="text-muted-foreground">Name</span>
                      <Input
                        value={voiceName}
                        onChange={(e) => setVoiceName(e.target.value)}
                        className="h-8"
                        data-testid="clone-voice-name"
                      />
                    </label>
                    <Button size="sm" variant="outline" onClick={() => void saveVoice()} data-testid="clone-save">
                      <Save /> {ref.savedId ? 'Update saved voice' : 'Save in this browser'}
                    </Button>
                  </div>
                </section>
              )}
            </div>
          </Panel>

          <Panel title="2. What it should say" data-testid="clone-speak">
            <div className="grid gap-3">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                aria-label="Text to speak"
                data-testid="clone-text"
              />
              <div className="flex flex-wrap items-center gap-2">
                {model.languages.length > 1 && (
                  <label className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Language</span>
                    <Select value={textLanguage} onValueChange={setLanguage}>
                      <SelectTrigger
                        className="h-7 w-36"
                        aria-label="Language of the text"
                        data-testid="clone-language"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent position="popper" className="max-h-72">
                        {model.languages.map((l) => (
                          <SelectItem key={l} value={l}>
                            {LANGUAGE_NAMES[l] ?? l}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                )}
                {s.busy ? (
                  <Button size="sm" variant="destructive" onClick={() => cloner.cancel()} data-testid="clone-stop">
                    <Square /> Stop
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    onClick={() => void generate()}
                    disabled={!ref || !text.trim() || busy}
                    data-testid="clone-generate"
                  >
                    {busy ? <Loader2 className="animate-spin" /> : <Play />} Generate
                  </Button>
                )}
                {!ref && <span className="text-xs text-muted-foreground">Choose a voice first.</span>}
              </div>
              <p className="text-xs text-muted-foreground" data-testid="clone-status" aria-live="polite">
                {s.status === 'loading'
                  ? `Loading ${model.label}: ${s.stage ?? 'downloading'} · ${Math.round(100 * s.progress)}%`
                  : s.busy
                    ? 'Generating…'
                    : s.status === 'ready'
                      ? `${model.label} ready on ${s.device === 'webgpu' ? 'WebGPU' : 'the CPU'} (loaded in ${((s.loadMs ?? 0) / 1000).toFixed(1)} s)`
                      : s.status === 'error'
                        ? `Could not load: ${s.error}`
                        : `${model.label} downloads on first use (${fmtBytes(model.bytes)}).`}
              </p>
              {s.status === 'error' && s.outOfMemory && (
                <Alert variant="destructive" data-testid="clone-oom">
                  <TriangleAlert />
                  <AlertTitle>The browser ran out of memory for this model</AlertTitle>
                  <AlertDescription>
                    The other models in this page were unloaded and it was tried again on one CPU thread. Close other
                    tabs, or choose a smaller model (Pocket TTS works on phones).
                  </AlertDescription>
                </Alert>
              )}
              {output && (
                <div className="grid gap-2 rounded-lg border border-border p-3" data-testid="clone-output">
                  <audio controls src={output.url} className="h-8 w-full" aria-label="The generated speech" />
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4" data-testid="clone-metrics">
                    <div>
                      <dt className="text-muted-foreground">First audio</dt>
                      <dd className="num" data-testid="clone-ttfa">
                        {output.result.firstAudioMs == null
                          ? '—'
                          : `${(output.result.firstAudioMs / 1000).toFixed(2)} s`}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-muted-foreground">Real-time factor</dt>
                      <dd className="num" data-testid="clone-rtf" data-rtf={output.result.rtf.toFixed(3)}>
                        {output.result.rtf.toFixed(2)}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-muted-foreground">Audio</dt>
                      <dd className="num">{(output.result.audio.length / CLONE_RATE).toFixed(1)} s</dd>
                    </div>
                    <div>
                      <dt className="text-muted-foreground">Voice taken in</dt>
                      <dd className="num">
                        {output.result.voiceMs ? `${(output.result.voiceMs / 1000).toFixed(2)} s` : 'already'}
                      </dd>
                    </div>
                  </dl>
                  <p className="text-[11px] text-muted-foreground">
                    Real-time factor: time spent generating divided by the length of the audio (under 1 keeps up with
                    playback).{' '}
                    {output.model.streams ? 'Played as it was made.' : 'This model makes each sentence whole.'}
                    {output.result.cancelled && ' Stopped early.'}
                  </p>
                  <SimilarityView s={output.similarity} />
                  {output.roundTrip && output.roundTrip !== 'busy' && (
                    <p className="text-xs" data-testid="clone-roundtrip" data-wer={output.roundTrip.wer.toFixed(3)}>
                      <span className="font-medium">Heard back ({output.roundTrip.model}):</span> “
                      {output.roundTrip.text}” · word error rate {(100 * output.roundTrip.wer).toFixed(0)}%
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <Button size="sm" variant="outline" asChild>
                      <a href={output.url} download={`bolo-clone-${output.model.id}.wav`} data-testid="clone-download">
                        <Download /> Download WAV
                      </a>
                    </Button>
                    {output.similarity == null && ref && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void score(output, ref)}
                        data-testid="clone-score"
                      >
                        Score similarity
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void roundTrip(output)}
                      disabled={output.roundTrip === 'busy'}
                      data-testid="clone-check"
                    >
                      {output.roundTrip === 'busy' ? <Loader2 className="animate-spin" /> : null} Check the words
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </Panel>
        </div>

        <div className="grid min-w-0 content-start gap-4">
          <Panel title="Model" description="Runs in this page; downloaded on first use" data-testid="clone-model">
            <div className="grid gap-3 text-xs">
              <Select value={model.id} onValueChange={(id) => cloner.setModel(id)} disabled={busy}>
                <SelectTrigger
                  className="h-auto min-h-8 w-full py-1"
                  aria-label="Cloning model"
                  data-testid="clone-model-select"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent position="popper" className="w-[min(26rem,calc(100vw-2rem))]">
                  {grouped.map(({ g, models }) => (
                    <SelectGroup key={g}>
                      <SelectLabel>{g}</SelectLabel>
                      {models.map((m) => (
                        <SelectItem key={m.id} value={m.id} textValue={m.label} data-testid={`clone-model-${m.id}`}>
                          <ModelOption m={m} d={d} />
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
              <VerdictLine verdict={verdict} bytes={model.bytes} />
              <dl className="grid gap-1.5" data-testid="clone-model-facts">
                <Fact k="Size">
                  {fmtBytes(model.bytes)} download, about {fmtBytes(model.peakBytes)} in memory
                </Fact>
                <Fact k="Licence">
                  <a
                    className="text-link underline underline-offset-2"
                    href={model.source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {model.license}
                  </a>
                  {model.licenseNote ? ` · ${model.licenseNote}` : ''}
                </Fact>
                <Fact k="Languages">{model.languages.map((l) => LANGUAGE_NAMES[l] ?? l).join(', ')}</Fact>
                <Fact k="Needs">
                  {model.needs ??
                    (model.prefersWebGpu
                      ? 'WebGPU for usable speed (a laptop or desktop); it runs on the CPU, many times slower than real time'
                      : 'any recent browser; quick on the CPU, phones included')}
                </Fact>
                <Fact k="The voice">{model.reference}</Fact>
                {model.watermark && <Fact k="Watermark">{model.watermark}</Fact>}
              </dl>
              <p className="text-muted-foreground">{model.blurb}</p>
              <label className="flex items-center gap-2">
                <span className="text-muted-foreground">Run on</span>
                <Select value={s.placement} onValueChange={(p) => cloner.setPlacement(p as Placement)} disabled={busy}>
                  <SelectTrigger className="h-7 w-32" aria-label="Run the model on" data-testid="clone-placement">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto</SelectItem>
                    <SelectItem value="webgpu" disabled={d.webgpu?.available === false}>
                      WebGPU
                    </SelectItem>
                    <SelectItem value="wasm">CPU</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              <label className="flex items-start gap-2" htmlFor="clone-score-auto">
                <Checkbox
                  id="clone-score-auto"
                  checked={scoreAuto}
                  onCheckedChange={(v) => setScoreAuto(v === true)}
                  className="mt-0.5"
                />
                <span>
                  Score speaker similarity after each clip ({SIMILARITY_MODEL.label}, {fmtBytes(SIMILARITY_MODEL.bytes)}
                  , {SIMILARITY_MODEL.license})
                </span>
              </label>
              {s.status !== 'idle' && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => cloner.dispose()}
                  disabled={s.busy}
                  data-testid="clone-unload"
                >
                  Unload the model
                </Button>
              )}
            </div>
          </Panel>
          <Panel title="Not offered here" description="From the shortlist, and why" data-testid="clone-not-offered">
            <ul className="grid gap-2 text-xs">
              {NOT_OFFERED.map((m) => (
                <li key={m.label}>
                  <span className="font-medium">{m.label}</span>{' '}
                  <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                    {m.license}
                  </Badge>
                  <span className="text-muted-foreground">
                    {' '}
                    · {m.size}. {m.why}
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
          <DownloadsPanel />
        </div>
      </div>
      <RiskDialog
        open={risk}
        label={model.label}
        bytes={model.bytes}
        verdict={verdict}
        onCancel={() => setRisk(false)}
        onConfirm={() => {
          setRisk(false)
          void generate(true)
        }}
      />
    </Page>
  )
}

function Fact({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

/** The chosen reference as a playable clip. */
function RefPlayer({ audio }: { audio: Float32Array }) {
  const url = useMemo(
    () => URL.createObjectURL(new Blob([encodeWav(audio, CLONE_RATE) as BlobPart], { type: 'audio/wav' })),
    [audio],
  )
  useEffect(() => () => URL.revokeObjectURL(url), [url])
  return (
    <audio controls src={url} className="h-8 w-full" aria-label="Play the chosen voice" data-testid="clone-ref-audio" />
  )
}
