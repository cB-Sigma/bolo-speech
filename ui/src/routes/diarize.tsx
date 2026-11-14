import { AudioLines, FileUp, Loader2, Mic, Play, Square, Users } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { DownloadsPanel } from '@/components/downloads-panel'
import { Page } from '@/components/layout/app-shell'
import { useDevice, VerdictBadge } from '@/components/model-fit'
import { Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { openMicrophone, type AudioSource } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import { sharedDiarizer } from '@/lib/browser-diarize/engine'
import {
  decodeTo16k,
  EXAMPLE_IDS,
  exampleAudioUrl,
  loadExample,
  recogniserFor,
  speakerName,
  turnsToTranscribe,
  type Example,
} from '@/lib/browser-diarize/examples'
import { EMBEDDINGS, SEGMENTATION, SERVER_DIARIZERS } from '@/lib/browser-diarize/models'
import { der, type DerResult, type Segment } from '@/lib/browser-diarize/pipeline'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { browserModel } from '@/lib/browser-stt/models'
import { peakFor, verdictFor } from '@/lib/device'
import { isDownloaded } from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

const COLORS = ['bg-chart-1', 'bg-chart-2', 'bg-chart-3', 'bg-chart-4', 'bg-chart-5', 'bg-chart-1/50']
const THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]

interface Lane {
  name: string
  segments: { start: number; end: number }[]
}

/** Who spoke when: one lane per speaker on a shared time axis. */
function Timeline({
  lanes,
  seconds,
  label,
  testId,
}: {
  lanes: Lane[]
  seconds: number
  label: string
  testId: string
}) {
  const total = Math.max(seconds, 0.1)
  return (
    <figure className="grid gap-1" data-testid={testId} aria-label={label}>
      <figcaption className="text-xs text-muted-foreground">{label}</figcaption>
      {lanes.map((lane, i) => (
        <div key={lane.name} className="flex items-center gap-2" data-testid="lane" data-speaker={lane.name}>
          <span className="w-8 shrink-0 font-mono text-xs">{lane.name}</span>
          <div className="relative h-4 min-w-0 flex-1 rounded bg-muted">
            {lane.segments.map((s, j) => (
              <span
                key={j}
                className={cn('absolute inset-y-0 rounded-sm', COLORS[i % COLORS.length])}
                style={{
                  left: `${(100 * s.start) / total}%`,
                  width: `${Math.max(0.3, (100 * (s.end - s.start)) / total)}%`,
                }}
                title={`${lane.name}: ${s.start.toFixed(1)}–${s.end.toFixed(1)} s`}
              />
            ))}
          </div>
        </div>
      ))}
      <div className="flex justify-between pl-10 font-mono text-[10px] text-muted-foreground">
        <span>0 s</span>
        <span>{(total / 2).toFixed(0)} s</span>
        <span>{total.toFixed(0)} s</span>
      </div>
    </figure>
  )
}

function lanesOf(segments: Segment[], names: (i: number) => string): Lane[] {
  const ids = [...new Set(segments.map((s) => s.speaker))].sort((a, b) => a - b)
  return ids.map((id) => ({ name: names(id), segments: segments.filter((s) => s.speaker === id) }))
}

interface RunResult {
  segments: Segment[]
  ms: number
  seconds: number
  score: DerResult | null
  transcript: { speaker: number; start: number; end: number; text: string }[] | null
}

/** Run the diarizer on 16 kHz audio with the page's settings. */
function useDiarize() {
  const [busy, setBusy] = useState<string | null>(null)
  const run = async (
    key: string,
    audio: Float32Array,
    opts: { speakers: number | null; threshold: number },
    reference?: Example,
  ): Promise<RunResult | null> => {
    setBusy(key)
    try {
      const r = await sharedDiarizer().run(audio, opts)
      const score = reference
        ? der(
            reference.turns,
            r.segments.map((s) => ({ speaker: speakerName(s.speaker), start: s.start, end: s.end })),
          )
        : null
      return { ...r, seconds: audio.length / 16000, score, transcript: null }
    } catch (err) {
      toast.error('Diarization failed', { description: err instanceof Error ? err.message : String(err) })
      return null
    } finally {
      setBusy(null)
    }
  }
  return { busy, run }
}

/** One speaker's lines, transcribed in the page with a recogniser for the language. */
async function transcribeTurns(audio: Float32Array, segments: Segment[], language: string) {
  const engine = sharedEngine()
  const model = recogniserFor(language, engine.store.get().model.id)
  engine.setModel(model)
  const out: RunResult['transcript'] = []
  for (const t of turnsToTranscribe(segments)) {
    const clip = audio.subarray(Math.floor(t.start * 16000), Math.ceil(t.end * 16000))
    if (clip.length < 16000 * 0.3) continue
    const r = await engine.transcribe(clip, { language })
    out.push({ ...t, text: r.text.trim() })
  }
  return { transcript: out, model }
}

function ResultView({
  result,
  reference,
  audio,
  language,
  onTranscript,
}: {
  result: RunResult
  reference?: Example
  audio: Float32Array
  language: string
  onTranscript: (t: RunResult['transcript']) => void
}) {
  const [busy, setBusy] = useState(false)
  const [model, setModel] = useState<string | null>(null)
  const map = result.score?.mapping ?? {}
  const name = (i: number) => {
    const s = speakerName(i)
    return map[s] ? `${s}` : s
  }
  const transcribe = async () => {
    setBusy(true)
    try {
      const r = await transcribeTurns(audio, result.segments, language)
      setModel(r.model)
      onTranscript(r.transcript)
    } catch (err) {
      toast.error('Transcription failed', { description: err instanceof Error ? err.message : String(err) })
    } finally {
      setBusy(false)
    }
  }
  const speakers = new Set(result.segments.map((s) => s.speaker)).size
  return (
    <div className="grid gap-3" data-testid="diarize-result">
      {reference && (
        <Timeline
          testId="reference-timeline"
          label="Reference: who really spoke when"
          seconds={result.seconds}
          lanes={reference.speakers.map((s) => ({ name: s, segments: reference.turns.filter((t) => t.speaker === s) }))}
        />
      )}
      <Timeline
        testId="result-timeline"
        label={`Result: ${speakers} speaker${speakers === 1 ? '' : 's'} found in ${(result.ms / 1000).toFixed(1)} s (${(result.ms / 1000 / result.seconds).toFixed(2)}× real time)`}
        seconds={result.seconds}
        lanes={lanesOf(result.segments, name)}
      />
      {result.score && (
        <p className="text-xs" data-testid="der">
          <span className="font-medium">DER {(100 * result.score.der).toFixed(1)}%</span>
          <span className="text-muted-foreground">
            {' '}
            · missed {result.score.missed.toFixed(1)} s · false alarm {result.score.falseAlarm.toFixed(1)} s · confusion{' '}
            {result.score.confusion.toFixed(1)} s of {result.score.total.toFixed(1)} s of speech (10 ms frames, no
            collar, overlaps scored) ·{' '}
            {Object.entries(map)
              .map(([h, r]) => `${h} = ${r}`)
              .join(', ')}
          </span>
        </p>
      )}
      {result.transcript ? (
        <div className="grid gap-1.5" data-testid="speaker-transcript">
          <p className="text-xs text-muted-foreground">
            Each speaker&apos;s turns transcribed in this page with {browserModel(model).label}.
          </p>
          {result.transcript.map((t, i) => (
            <p key={i} className="text-[13px] leading-5">
              <span className="mr-2 inline-flex items-center gap-1 font-mono text-[11px] font-medium">
                <span className={cn('size-2 rounded-sm', COLORS[t.speaker % COLORS.length])} aria-hidden />
                {name(t.speaker)}
              </span>
              <span className="mr-2 font-mono text-[11px] text-muted-foreground">{t.start.toFixed(1)} s</span>
              {t.text || <span className="text-muted-foreground">(nothing recognised)</span>}
            </p>
          ))}
        </div>
      ) : (
        <div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void transcribe()}
            disabled={busy}
            data-testid="transcribe-speakers"
          >
            {busy ? <Loader2 className="animate-spin" /> : <AudioLines />} Transcript per speaker
          </Button>
        </div>
      )}
    </div>
  )
}

function ExampleCard({
  id,
  opts,
  diarize,
}: {
  id: string
  opts: { speakers: number | null; threshold: number }
  diarize: ReturnType<typeof useDiarize>
}) {
  const [ex, setEx] = useState<Example | null>(null)
  const [result, setResult] = useState<RunResult | null>(null)
  const [audio, setAudio] = useState<Float32Array | null>(null)
  useEffect(() => {
    loadExample(id).then(setEx, () => undefined)
  }, [id])
  const run = async () => {
    if (!ex) return
    const a = audio ?? (await decodeTo16k(await (await fetch(exampleAudioUrl(id))).arrayBuffer()))
    setAudio(a)
    const r = await diarize.run(id, a, opts, ex)
    if (r) setResult(r)
  }
  return (
    <li className="grid min-w-0 gap-2 rounded-lg border border-border p-3" data-testid={`example-${id}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <h3 className="text-sm font-medium">{ex?.title ?? id}</h3>
        {ex && (
          <span className="text-xs text-muted-foreground">
            {ex.seconds.toFixed(0)} s · {ex.speakers.length} speakers · {ex.language === 'hi' ? 'Hindi' : 'English'}
          </span>
        )}
      </div>
      <audio
        controls
        preload="none"
        src={exampleAudioUrl(id)}
        className="h-8 w-full"
        aria-label={`Play: ${ex?.title ?? id}`}
      />
      <div>
        <Button
          size="sm"
          onClick={() => void run()}
          disabled={!ex || diarize.busy != null}
          data-testid={`run-example-${id}`}
        >
          {diarize.busy === id ? <Loader2 className="animate-spin" /> : <Users />} Run diarization on this example
        </Button>
      </div>
      {result && audio && ex && (
        <ResultView
          result={result}
          reference={ex}
          audio={audio}
          language={ex.language}
          onTranscript={(t) => setResult({ ...result, transcript: t })}
        />
      )}
      {ex && <p className="text-[11px] leading-4 text-muted-foreground">Voices: {ex.credits.join('; ')}.</p>}
    </li>
  )
}

export default function DiarizePage() {
  const diarizer = sharedDiarizer()
  const s = useStore(diarizer.store)
  const d = useDevice()
  const [count, setCount] = useState<string>('auto')
  const [threshold, setThreshold] = useState(0.6)
  const [language, setLanguage] = useState('en')
  const [own, setOwn] = useState<{ audio: Float32Array; name: string } | null>(null)
  const [ownResult, setOwnResult] = useState<RunResult | null>(null)
  const [mic, setMic] = useState<AudioSource | null>(null)
  const blocks = useRef<Float32Array[]>([])
  const diarize = useDiarize()
  const opts = { speakers: count === 'auto' ? null : Number(count), threshold }

  const upload = async (file: File) => {
    try {
      const audio = await decodeTo16k(await file.arrayBuffer())
      setOwn({ audio, name: file.name })
      setOwnResult(null)
    } catch {
      toast.error('This browser cannot read that file', { description: 'Try WAV, MP3, M4A, OGG or WebM.' })
    }
  }
  const record = async () => {
    if (mic) {
      await mic.stop()
      setMic(null)
      setOwn({ audio: concatFloat(blocks.current), name: 'recording' })
      setOwnResult(null)
      return
    }
    blocks.current = []
    try {
      setMic(await openMicrophone((b) => blocks.current.push(b.slice())))
    } catch (err) {
      toast.error('No microphone', { description: err instanceof Error ? err.message : String(err) })
    }
  }

  return (
    <Page title="Speakers">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="grid min-w-0 gap-4">
          <Panel
            title="Ready examples"
            description="Conversations made with the demo's own voices, with the real who-spoke-when to score against"
            data-testid="diarize-examples"
          >
            <ul className="grid gap-3">
              {EXAMPLE_IDS.map((id) => (
                <ExampleCard key={id} id={id} opts={opts} diarize={diarize} />
              ))}
            </ul>
          </Panel>

          <Panel
            title="Your audio"
            description="Upload a file or record; it stays in this page"
            data-testid="diarize-own"
          >
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="outline" asChild>
                <label className="cursor-pointer">
                  <FileUp /> Upload audio
                  <input
                    type="file"
                    accept="audio/*,video/*"
                    className="sr-only"
                    data-testid="diarize-upload"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f) void upload(f)
                    }}
                  />
                </label>
              </Button>
              <Button
                size="sm"
                variant={mic ? 'destructive' : 'outline'}
                onClick={() => void record()}
                data-testid="diarize-record"
              >
                {mic ? <Square /> : <Mic />} {mic ? 'Stop recording' : 'Record'}
              </Button>
              <label className="flex items-center gap-2 text-xs">
                <span className="text-muted-foreground">Language for the transcript</span>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger className="h-7 w-28" aria-label="Language for the transcript">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="hi">Hindi</SelectItem>
                  </SelectContent>
                </Select>
              </label>
            </div>
            {own && (
              <div className="mt-3 grid gap-3">
                <p className="text-xs text-muted-foreground" data-testid="own-audio">
                  {own.name} · {(own.audio.length / 16000).toFixed(1)} s
                </p>
                <div>
                  <Button
                    size="sm"
                    onClick={() =>
                      void diarize.run('own', own.audio, opts).then((r) => {
                        if (r) setOwnResult(r)
                      })
                    }
                    disabled={diarize.busy != null}
                    data-testid="run-own"
                  >
                    {diarize.busy === 'own' ? <Loader2 className="animate-spin" /> : <Play />} Who spoke when
                  </Button>
                </div>
                {ownResult && (
                  <ResultView
                    result={ownResult}
                    audio={own.audio}
                    language={language}
                    onTranscript={(t) => setOwnResult({ ...ownResult, transcript: t })}
                  />
                )}
              </div>
            )}
          </Panel>
        </div>

        <div className="grid min-w-0 content-start gap-4">
          <Panel
            title="Settings"
            description="Runs in this page with ONNX Runtime Web (WebAssembly)"
            data-testid="diarize-settings"
          >
            <div className="grid gap-3 text-xs">
              <label className="grid gap-1">
                <span className="text-muted-foreground">Voice embedding model</span>
                <Select
                  value={s.embedding.id}
                  onValueChange={(id) => diarizer.setEmbedding(id)}
                  disabled={diarize.busy != null}
                >
                  <SelectTrigger
                    className="h-8 w-full"
                    aria-label="Voice embedding model"
                    data-testid="embedding-select"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" className="w-[min(24rem,calc(100vw-2rem))]">
                    {EMBEDDINGS.map((m) => (
                      <SelectItem key={m.id} value={m.id} textValue={m.label} data-testid={`embedding-${m.id}`}>
                        <span className="grid">
                          <span>{m.label}</span>
                          <span className="text-[11px] text-muted-foreground">
                            {fmtBytes(m.bytes)} · {m.license}
                            {m.heavy ? ` · ${m.heavy.split(':')[0]}` : ''}
                          </span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>
              <div className="flex flex-wrap items-center gap-2" data-testid="embedding-about">
                <VerdictBadge
                  verdict={verdictFor(
                    {
                      bytes: s.embedding.bytes + SEGMENTATION.bytes,
                      peakBytes: peakFor(`diarize-${s.embedding.id}`, s.embedding.peakBytes + SEGMENTATION.peakBytes),
                      downloaded: isDownloaded(`diarize-${s.embedding.id}`),
                    },
                    d,
                  )}
                />
                <span className="text-muted-foreground">
                  {s.embedding.blurb} {s.embedding.licenseNote ?? ''}{' '}
                  {s.embedding.heavy ? `Best on a ${s.embedding.heavy}.` : ''}
                </span>
              </div>
              <p className="text-muted-foreground">
                With {SEGMENTATION.label} ({fmtBytes(SEGMENTATION.bytes)}, {SEGMENTATION.license}): {SEGMENTATION.blurb}
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <label className="flex items-center gap-2">
                  <span className="text-muted-foreground">Speakers</span>
                  <Select value={count} onValueChange={setCount}>
                    <SelectTrigger className="h-7 w-24" aria-label="Number of speakers" data-testid="speaker-count">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Automatic</SelectItem>
                      {[1, 2, 3, 4, 5, 6].map((n) => (
                        <SelectItem key={n} value={String(n)}>
                          {n}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </label>
                {count === 'auto' && (
                  <label className="flex items-center gap-2">
                    <span className="text-muted-foreground">Threshold</span>
                    <Select value={String(threshold)} onValueChange={(v) => setThreshold(Number(v))}>
                      <SelectTrigger className="h-7 w-20" aria-label="Clustering threshold" data-testid="threshold">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {THRESHOLDS.map((t) => (
                          <SelectItem key={t} value={String(t)}>
                            {t}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                )}
              </div>
              <p className="text-muted-foreground">
                Automatic: voices closer than the threshold (cosine distance) count as one speaker; lower finds more
                speakers.
              </p>
              <p data-testid="diarizer-status">
                {s.status === 'loading'
                  ? `Downloading the models · ${Math.round(100 * s.progress)}%`
                  : s.status === 'ready'
                    ? s.running != null
                      ? `Working · ${Math.round(100 * s.running)}%`
                      : `Ready (loaded in ${((s.loadMs ?? 0) / 1000).toFixed(1)} s)`
                    : s.status === 'error'
                      ? `Could not load: ${s.error}`
                      : 'Downloaded on first use'}
              </p>
            </div>
          </Panel>
          <Panel
            title="Heavier diarizers"
            description="What a bolo server runs that this page does not"
            data-testid="diarize-server"
          >
            <ul className="grid gap-2 text-xs">
              {SERVER_DIARIZERS.map((m) => (
                <li key={m.label}>
                  <a
                    className="font-medium text-link underline underline-offset-2"
                    href={m.source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {m.label}
                  </a>{' '}
                  <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                    {m.license}
                  </Badge>
                  <span className="text-muted-foreground">
                    {' '}
                    · {m.size} · {m.needs}
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
          <DownloadsPanel />
        </div>
      </div>
    </Page>
  )
}
