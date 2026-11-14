import { Download, ExternalLink, Info, Languages, Play, Repeat2, Square, Trash2, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { BudgetBar } from '@/components/charts/budget-bar'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel, Readout } from '@/components/panel'
import { LoadProgress, LoadTrouble, ModelStatusChip } from '@/components/stt/model-status'
import { ChunkTimeline } from '@/components/tts/chunk-timeline'
import { PlaybackWave } from '@/components/tts/playback-wave'
import { RoundTripPanel, type RoundTrip } from '@/components/tts/round-trip'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useStore } from '@/hooks/use-store'
import { isDemo } from '@/lib/api/config'
import { STT_RATE } from '@/lib/audio/capture'
import { resample } from '@/lib/audio/resample'
import { encodeWav } from '@/lib/audio/wav'
import { download } from '@/lib/bench'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { browserModel } from '@/lib/browser-stt/models'
import { clearVoiceCache, sharedTtsEngine } from '@/lib/browser-tts/engine'
import { BrowserTtsSession, type BrowserTtsExtra } from '@/lib/browser-tts/session'
import {
  browserVoice,
  BROWSER_VOICES,
  languageLabel,
  voicesByLanguage,
  type BrowserVoice,
} from '@/lib/browser-tts/voices'
import { fmtBytes, fmtMs, fmtRtf, fmtSeconds, fmtSpeed } from '@/lib/format'
import { wordErrorRate } from '@/lib/latency/wer'
import { prefs } from '@/lib/prefs'
import { Store } from '@/lib/store'
import { emptyTts, type TtsState } from '@/lib/tts/session'
import { cn } from '@/lib/utils'

const idle = new Store<TtsState>(emptyTts())
const idleExtra = new Store<BrowserTtsExtra>({
  prepared: null,
  loadWaitMs: null,
  computeMs: 0,
  phonemes: [],
  missing: [],
})
const SAMPLES = new Set(BROWSER_VOICES.map((v) => v.sample))
const VOICE_PREF = 'bolo.browserTtsVoice'

function savedVoice(): BrowserVoice {
  try {
    return browserVoice(globalThis.localStorage?.getItem(VOICE_PREF))
  } catch {
    return browserVoice(null)
  }
}

/** HTML `lang` for a voice's text (so the right font and screen-reader voice are used). */
function htmlLang(v: BrowserVoice): string {
  return v.language
}

function VoicePicker({
  value,
  onChange,
  disabled,
  sizes,
}: {
  value: BrowserVoice
  onChange: (id: string) => void
  disabled?: boolean
  sizes: Record<string, number>
}) {
  return (
    <Select value={value.id} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="h-8 w-64 max-sm:min-w-0 max-sm:flex-1" aria-label="Voice" data-testid="voice-picker">
        <span className="min-w-0 truncate text-left">
          <span lang={htmlLang(value)}>{value.nativeName}</span>
          {value.nativeName !== value.englishName && !value.englishName.startsWith(value.nativeName) && (
            <span className="text-muted-foreground"> · {value.englishName}</span>
          )}
          <span className="text-muted-foreground"> · {value.gender}</span>
        </span>
      </SelectTrigger>
      <SelectContent position="popper" align="start" className="max-h-96">
        {voicesByLanguage().map((g) => (
          <SelectGroup key={g.englishName}>
            <SelectLabel>
              <span lang={g.language}>{languageLabel(g)}</span>
            </SelectLabel>
            {g.voices.map((v) => (
              <SelectItem key={v.id} value={v.id} data-testid={`voice-${v.id}`}>
                <span>{v.label}</span>
                <span className="ml-2 num text-xs text-muted-foreground">{fmtBytes(sizes[v.id] ?? v.approxBytes)}</span>
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  )
}

function AboutVoice({ voice, bytes, onCleared }: { voice: BrowserVoice; bytes: number; onCleared: () => void }) {
  return (
    <Panel title="About this voice" bodyClassName="grid gap-3 text-[13px] leading-5" data-testid="voice-about">
      <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-1.5">
        <dt className="text-muted-foreground">Language</dt>
        <dd lang={voice.language}>{languageLabel(voice)}</dd>
        <dt className="text-muted-foreground">Made by</dt>
        <dd>{voice.credit}</dd>
        <dt className="text-muted-foreground">Licence</dt>
        <dd>
          Weights {voice.license}; training data {voice.dataLicense}
        </dd>
        <dt className="text-muted-foreground">Download</dt>
        <dd className="num">
          {fmtBytes(bytes)}
          <span className="font-sans text-muted-foreground"> · once, then kept by the browser</span>
        </dd>
        <dt className="text-muted-foreground">Source</dt>
        <dd className="min-w-0">
          <a
            href={voice.source}
            target="_blank"
            rel="noreferrer"
            className="inline-flex max-w-full items-center gap-1 break-all underline decoration-border underline-offset-2 hover:decoration-foreground"
          >
            {voice.source.replace(/^https:\/\//, '')}
            <ExternalLink className="size-3 shrink-0" aria-hidden />
          </a>
        </dd>
      </dl>
      <ul className="grid list-disc gap-1.5 pl-4 text-muted-foreground">
        {voice.notes.map((n) => (
          <li key={n}>{n}</li>
        ))}
        <li>Nobody who speaks this language has reviewed what this page makes the voice say.</li>
        <li>
          Converted for the browser by bolo: 8-bit weights where that made no measurable difference, 16-bit elsewhere.
          The voice was not retrained.
        </li>
      </ul>
      <div>
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            void clearVoiceCache()
              .then(onCleared)
              .then(() => toast.success('Downloaded voices removed from this browser'))
          }
          data-testid="clear-voices"
        >
          <Trash2 /> Remove downloaded voices
        </Button>
      </div>
    </Panel>
  )
}

/**
 * Text to speech in the page: the voice is downloaded once, runs in a worker
 * with ONNX Runtime Web, and each sentence plays as soon as it is ready.
 */
export function BrowserTts({ modeToggle }: { modeToggle?: ReactNode }) {
  const engine = sharedTtsEngine()
  const es = useStore(engine.store)
  const p = useStore(prefs)
  const [voice, setVoice] = useState<BrowserVoice>(savedVoice)
  const [text, setText] = useState(voice.sample)
  const [session, setSession] = useState<BrowserTtsSession | null>(null)
  const [trip, setTrip] = useState<RoundTrip | null>(null)
  const state = useStore(session?.store ?? idle)
  const extra = useStore(session?.extra ?? idleExtra)
  const busy = state.status === 'connecting' || state.status === 'synthesizing' || state.status === 'playing'
  const tripRef = useRef(0)

  useEffect(() => {
    engine.setVoice(voice.id)
    void engine.readIndex()
  }, [engine, voice.id])
  useEffect(() => () => void session?.dispose(), [session])
  useEffect(() => {
    if (state.status === 'error' && state.error) toast.error('Synthesis failed', { description: state.error })
  }, [state.status, state.error])

  const sizes = Object.fromEntries((es.index?.voices ?? []).map((v) => [v.id, v.bytes]))
  const published = es.index ? new Set(es.index.voices.map((v) => v.id)) : null
  const unavailable = published != null && !published.has(voice.id)

  const chooseVoice = (id: string) => {
    const next = browserVoice(id)
    session?.stop()
    setVoice(next)
    setTrip(null)
    try {
      globalThis.localStorage?.setItem(VOICE_PREF, next.id)
    } catch {
      // ignore
    }
    // Keep what the user typed; swap a sample sentence for the new voice's own.
    if (!text.trim() || SAMPLES.has(text)) setText(next.sample)
  }

  const speak = (): BrowserTtsSession | null => {
    if (!text.trim() || unavailable) return null
    session?.stop()
    const s = new BrowserTtsSession(engine, voice, text.trim())
    setSession(s)
    setTrip(null)
    void s.start()
    return s
  }

  const save = () => {
    if (!session || !state.sampleRate) return
    download(`bolo-${voice.id}.wav`, encodeWav(session.audio(), state.sampleRate), 'audio/wav')
  }

  /** Each synthesised sentence through the in-browser recogniser, scored against what was spoken. */
  const transcribe = async (s: BrowserTtsSession) => {
    const rt = voice.roundTrip
    if (!rt) return
    const token = ++tripRef.current
    const stt = sharedEngine()
    const label = browserModel(rt.model).label
    setTrip({ status: 'running', model: label, hypothesis: '', result: null, where: 'in this browser' })
    try {
      await new Promise<void>((resolve, reject) => {
        const check = () => {
          const st = s.store.get()
          if (st.status === 'error') reject(new Error(st.error ?? 'Synthesis failed'))
          else if (st.totalMs != null) resolve()
          else return false
          return true
        }
        if (check()) return
        const off = s.store.subscribe(() => check() && off())
      })
      // Borrow the recogniser for this check only: the model chosen on the speech-to-text screen stays chosen.
      if (stt.store.get().model.id !== rt.model) stt.setModel(rt.model)
      const st = s.store.get()
      const audio = s.audio()
      const rate = st.sampleRate ?? 22050
      const parts: string[] = []
      // One sentence at a time: the recognisers read at most 30 seconds per pass.
      for (const c of st.chunks) {
        const slice = audio.subarray(Math.round(c.t0 * rate), Math.round((c.t0 + c.dur) * rate))
        const r = await stt.transcribe(resample(slice, rate, STT_RATE), { language: rt.language })
        parts.push(r.text)
      }
      if (token !== tripRef.current) return
      const hypothesis = parts.join(' ').trim()
      const reference = s.extra.get().prepared?.spoken ?? text
      setTrip({
        status: 'done',
        model: label,
        hypothesis,
        result: wordErrorRate(reference, hypothesis),
        where: 'in this browser',
      })
    } catch (err) {
      if (token !== tripRef.current) return
      setTrip({
        status: 'error',
        model: label,
        hypothesis: '',
        result: null,
        error: err instanceof Error ? err.message : String(err),
        where: 'in this browser',
      })
    }
  }

  const roundTrip = () => {
    const s = state.totalMs != null && session && !busy ? session : speak()
    if (s) void transcribe(s)
  }

  const hasAudio = state.audioSeconds > 0 && state.totalMs != null
  const underrun = state.underruns > 0
  const bytes = sizes[voice.id] ?? voice.approxBytes
  const prepared = extra.prepared
  const tripTip = voice.roundTrip
    ? `Transcribes the audio with ${browserModel(voice.roundTrip.model).label} in this browser (${fmtBytes(browserModel(voice.roundTrip.model).approxBytes)} the first time)`
    : `There is no in-browser recogniser for ${voice.englishName} yet`

  const toolbar = (
    <>
      <VoicePicker value={voice} onChange={chooseVoice} disabled={busy} sizes={sizes} />
      {modeToggle && <div className="ml-auto">{modeToggle}</div>}
    </>
  )

  return (
    <Page title="Text to speech" toolbar={toolbar}>
      {es.indexError && (
        <Alert className="mb-4" data-testid="voices-missing">
          <Info />
          <AlertTitle>The voices could not be found</AlertTitle>
          <AlertDescription>
            {isDemo
              ? `This copy of the demo was built without its voices (${es.indexError}).`
              : `In-browser voices are fetched from the public demo, which could not be reached (${es.indexError}). Server mode is unaffected.`}
          </AlertDescription>
        </Alert>
      )}
      {unavailable && (
        <Alert className="mb-4">
          <Info />
          <AlertTitle>{voice.label} is not published with this build</AlertTitle>
          <AlertDescription>Pick another voice.</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="Text"
            description={`${text.trim().length} characters · ${languageLabel(voice)}`}
            actions={<ModelStatusChip engine={engine} noun="voice" />}
          >
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              aria-label="Text to speak"
              placeholder="Type something to say…"
              className="min-h-28 resize-y text-[15px] leading-7"
              lang={htmlLang(voice)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !busy) speak()
              }}
              data-testid="tts-text"
            />
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Languages className="mr-0.5 size-3.5 text-muted-foreground" aria-hidden />
              <span className="mr-1 text-xs text-muted-foreground">Try</span>
              {voicesByLanguage().map((g) => {
                const v = g.voices.find((x) => x.gender === voice.gender) ?? (g.voices[0] as BrowserVoice)
                const active = g.voices.some((x) => x.id === voice.id)
                return (
                  <Button
                    key={g.englishName}
                    size="xs"
                    variant={active ? 'secondary' : 'ghost'}
                    onClick={() => {
                      if (!active) chooseVoice(v.id)
                      setText(active ? voice.sample : v.sample)
                    }}
                    disabled={busy}
                    title={v.sample}
                    className="border border-border"
                  >
                    <span lang={g.language}>
                      {g.englishName.startsWith(g.nativeName) ? g.englishName : g.nativeName}
                    </span>
                  </Button>
                )
              })}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {busy ? (
                <Button
                  onClick={() => session?.stop()}
                  variant="destructive"
                  className="h-9 min-w-24"
                  data-testid="tts-stop"
                >
                  <Square className="fill-current" /> Stop
                </Button>
              ) : (
                <Button
                  onClick={speak}
                  className="h-9 min-w-24"
                  disabled={!text.trim() || unavailable}
                  data-testid="speak"
                >
                  <Play className="fill-current" /> Speak
                </Button>
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  {/* A wrapper keeps the tooltip working while the button is disabled. */}
                  <span tabIndex={voice.roundTrip ? undefined : 0}>
                    <Button
                      variant="outline"
                      className="h-9"
                      onClick={roundTrip}
                      disabled={!voice.roundTrip || !text.trim() || busy || unavailable || trip?.status === 'running'}
                      data-testid="round-trip"
                    >
                      <Repeat2 /> Speak, then transcribe
                    </Button>
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-64">{tripTip}</TooltipContent>
              </Tooltip>
              <Button variant="ghost" className="h-9" onClick={save} disabled={!hasAudio} data-testid="save-wav">
                <Download /> WAV
              </Button>
            </div>
            <LoadProgress className="mt-3" engine={engine} noun="voice" />
            <LoadTrouble className="mt-3" engine={engine} noun="voice" />
            {prepared && (prepared.rewritten || prepared.dropped.length > 0 || extra.missing.length > 0) && (
              <div
                className="mt-3 rounded-lg border border-border bg-background/60 px-3 py-2 text-[13px] leading-5"
                data-testid="spoken-text"
              >
                {prepared.rewritten && (
                  <p>
                    <span className="text-muted-foreground">Read as </span>
                    <span lang={htmlLang(voice)}>{prepared.spoken}</span>
                  </p>
                )}
                {prepared.dropped.length > 0 && (
                  <p className="mt-1 flex items-start gap-1.5 text-muted-foreground" data-testid="dropped">
                    <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning-text" aria-hidden />
                    <span>
                      Skipped, the voice has no symbol for:{' '}
                      <span className="num text-foreground">
                        {prepared.dropped.map((c) => (c === ' ' ? '␠' : c)).join(' ')}
                      </span>
                    </span>
                  </p>
                )}
                {extra.missing.length > 0 && (
                  <p className="mt-1 text-muted-foreground">
                    Phonemes the voice cannot say:{' '}
                    <span className="num text-foreground">{extra.missing.join(' ')}</span>
                  </p>
                )}
              </div>
            )}
            {extra.phonemes.length > 0 && (
              <details className="mt-3 text-[13px]">
                <summary className="cursor-pointer text-muted-foreground">Phonemes (uccharan)</summary>
                <p className="mt-1 num leading-6 break-words" data-testid="phonemes">
                  {extra.phonemes.join(' ')}
                </p>
              </details>
            )}
          </Panel>

          <Panel
            title="Audio"
            description={
              state.status === 'idle'
                ? 'Nothing synthesised yet'
                : `${state.status === 'connecting' ? 'Getting the voice ready' : state.status === 'synthesizing' ? 'Synthesising and playing' : state.status === 'playing' ? 'Playing' : state.status === 'error' ? 'Failed' : 'Done'} · ${fmtSeconds(state.audioSeconds)}${state.sampleRate ? ` · ${(state.sampleRate / 1000).toFixed(state.sampleRate % 1000 ? 2 : 0)} kHz` : ''}`
            }
            actions={
              state.status !== 'idle' && state.status !== 'connecting' ? (
                <span
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 text-[11px]',
                    underrun ? 'border-warning/50 text-foreground' : 'border-border text-muted-foreground',
                  )}
                  data-testid="underruns"
                >
                  {underrun ? (
                    <>
                      <TriangleAlert className="size-3 text-warning-text" aria-hidden />
                      {state.underruns} underrun{state.underruns === 1 ? '' : 's'} · waited{' '}
                      <span className="num">{fmtMs(state.starvedMs)}</span>
                    </>
                  ) : (
                    'No underruns'
                  )}
                </span>
              ) : undefined
            }
          >
            <div className="rounded-lg border border-border bg-background/60 px-2 py-1">
              <PlaybackWave session={session} />
            </div>
            {state.chunks.length > 0 ? (
              <>
                <ChunkTimeline chunks={state.chunks} className="mt-4" />
                {state.rtf != null && state.rtf > 1 && (
                  <p className="mt-3 text-xs text-muted-foreground" data-testid="slow-device">
                    This device made the audio more slowly than it plays (real-time factor {fmtRtf(state.rtf)}), so
                    playback had to wait between sentences. The download WAV has no gaps.
                  </p>
                )}
              </>
            ) : (
              <p className="mt-3 text-xs text-muted-foreground">
                The text is synthesised one sentence at a time (a very short one together with its neighbour). Each
                piece starts playing as soon as it is ready while the next one is being made; if playback ever catches
                up, the gap is counted as an underrun.
              </p>
            )}
          </Panel>

          {trip && <RoundTripPanel trip={trip} lang={htmlLang(voice)} />}
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Latency" description="Measured in this browser, from the moment the voice was ready">
            {state.status === 'idle' || (state.status === 'connecting' && state.ttfaMs == null) ? (
              <Empty title="No measurement yet" className="py-6">
                Press Speak. The first time, the voice is downloaded ({fmtBytes(bytes)}) and compiled; after that it
                loads from the browser&rsquo;s cache.
              </Empty>
            ) : (
              <>
                <div className="flex items-baseline gap-2">
                  <span className="text-5xl leading-none font-semibold tracking-tight" data-testid="ttfa">
                    {state.ttfaMs != null ? Math.round(state.ttfaMs) : '–'}
                  </span>
                  <span className="text-sm text-muted-foreground">ms to first audio</span>
                </div>
                <BudgetBar
                  className="mt-4"
                  measuredMs={state.ttfaMs}
                  targetMs={p.targetMs}
                  label="Time to first audio"
                />
                <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-border pt-4">
                  <Readout
                    label="Real-time factor"
                    value={fmtRtf(state.rtf)}
                    hint={state.rtf ? `${fmtSpeed(state.rtf)} real time` : 'when synthesis completes'}
                  />
                  <Readout
                    label="Audio"
                    value={state.audioSeconds.toFixed(2)}
                    unit="s"
                    hint={`in ${state.chunks.length} piece${state.chunks.length === 1 ? '' : 's'}`}
                  />
                  <Readout
                    label="Voice ready after"
                    value={extra.loadWaitMs != null ? (extra.loadWaitMs / 1000).toFixed(1) : '–'}
                    unit="s"
                    hint={
                      extra.loadWaitMs === 0
                        ? 'it was already loaded'
                        : extra.loadWaitMs != null
                          ? 'download (or cache), compile, warm-up'
                          : undefined
                    }
                  />
                  <Readout
                    label="Runs on"
                    value={es.status === 'ready' ? `${es.threads}` : '–'}
                    unit={es.status === 'ready' ? `thread${es.threads === 1 ? '' : 's'}` : undefined}
                    hint="WebAssembly, ONNX Runtime Web"
                  />
                </div>
              </>
            )}
          </Panel>

          <AboutVoice voice={voice} bytes={bytes} onCleared={() => void engine.refreshCached()} />

          <Panel title="Reading the numbers" bodyClassName="grid gap-2.5 text-[13px] leading-5 text-muted-foreground">
            <p>
              <span className="text-foreground">Time to first audio</span> is how long the first piece took to
              synthesise. It does not include the one-time download, which is shown separately.
            </p>
            <p>
              <span className="text-foreground">Real-time factor</span> is time spent in the model per second of audio.
              Below 1, each sentence is ready before the previous one has finished playing.
            </p>
            <p>Everything here runs on this device: the text never leaves the page.</p>
          </Panel>
        </div>
      </div>
    </Page>
  )
}
