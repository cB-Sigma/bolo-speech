import { oneRuntimeAtATime } from '@/lib/browser-runtime'
import { interrupted, isDownloaded } from '@/lib/downloads'
import {
  Cloud,
  Copy,
  FileAudio,
  Info,
  Mic,
  MicOff,
  Phone,
  Play,
  Settings2,
  Share2,
  Square,
  Upload,
  Volume2,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { BudgetBar } from '@/components/charts/budget-bar'
import { LatencyTimeline, UtteranceList } from '@/components/charts/latency-timeline'
import { LevelStrip } from '@/components/charts/level-strip'
import { SegmentedBar } from '@/components/charts/segmented-bar'
import { Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { Panel, Readout } from '@/components/panel'
import { DevicePanel } from '@/components/model-fit'
import { DownloadsPanel, InterruptedLoadNotice } from '@/components/downloads-panel'
import { BrowserEnginePanel } from '@/components/stt/engine-panel'
import { BrowserProcessingPanel } from '@/components/stt/browser-processing'
import { audioEngine } from '@/lib/browser-audio/engine'
import { resample } from '@/lib/audio/resample'
import { routeByLanguage, type LidDecision } from '@/lib/stt/routing'
import { BrowserModelPicker, LanguageSelect, SwitchModelDialog } from '@/components/stt/language-controls'
import { MicHelpPanel, MicStatusLine } from '@/components/stt/mic-help'
import { LoadTrouble, ModelStatusChip } from '@/components/stt/model-status'
import { Transcript } from '@/components/stt/transcript'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { describeError, useCatalog, useInfo, useSamples, useSelectedModel } from '@/hooks/use-api'
import { LanguageIdPanel } from '@/components/stt/language-id-panel'
import { type Clip } from '@/components/voices/voice-compare'
import { SpeakersPanel } from '@/components/stt/speakers-panel'
import { AlignPanel } from '@/components/align/align-panel'
import { DenoiseControl } from '@/components/stt/transcribe-options'
import { defaultDenoise, type Denoise } from '@/lib/stt-options'
import { usePing } from '@/hooks/use-ping'
import { useStore } from '@/hooks/use-store'
import { useSttSession, type EngineMode, type SessionInput } from '@/hooks/use-stt-session'
import { useSupport } from '@/hooks/use-support'
import { api } from '@/lib/api/client'
import { cloudNotice, isCloud, notInstalledHint } from '@/lib/capabilities'
import type { Sample } from '@/lib/api/types'
import { isDemo } from '@/lib/api/config'
import { STT_RATE } from '@/lib/audio/capture'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { browserModel, BROWSER_MODELS } from '@/lib/browser-stt/models'
import { prefersLessData } from '@/lib/diagnostics'
import { INDIC_LANGUAGES, supportsLanguage } from '@/lib/languages'
import type { MicFailure } from '@/lib/mic-help'
import { decodeShare, shareUrl } from '@/lib/share'
import { SERVER_ONLY_HINT } from '@/components/demo'
import { useSearchParams } from 'react-router-dom'
import { concatFloat } from '@/lib/audio/pcm'
import { simulateTelephony, TELEPHONY_RATE } from '@/lib/audio/telephony'
import { decodeAudio } from '@/lib/audio/wav'
import { fmtBytes, fmtMs, fmtRtf, fmtSeconds, fmtSpeed, languageName } from '@/lib/format'
import { breakdown, summarize } from '@/lib/latency/stats'
import { utteranceLatency, type Utterance } from '@/lib/latency/timeline'
import { pickBrowserModel, prefs, setPref } from '@/lib/prefs'
import { Store } from '@/lib/store'
import { emptyTranscript, transcriptText, type Transcriber, type TranscriptState } from '@/lib/stt/transcriber'
import { cn } from '@/lib/utils'

const idleStore = new Store<TranscriptState>(emptyTranscript())

function useTranscript(t: Transcriber | null): TranscriptState {
  return useStore(t?.store ?? idleStore)
}

function lastFinal(utterances: Utterance[]): Utterance | null {
  for (let i = utterances.length - 1; i >= 0; i--) {
    if ((utterances[i] as Utterance).finalAt != null) return utterances[i] as Utterance
  }
  return utterances[utterances.length - 1] ?? null
}

function sessionStats(s: TranscriptState) {
  const lats = s.utterances.filter((u) => u.finalAt != null).map(utteranceLatency)
  return {
    count: lats.length,
    firstPartial: summarize(lats.map((l) => l.firstPartialMs).filter((v): v is number => v != null)),
    final: summarize(lats.map((l) => l.finalMs).filter((v): v is number => v != null)),
  }
}

const STATUS_TEXT: Record<TranscriptState['status'], string> = {
  idle: 'Idle',
  loading: 'Loading model',
  connecting: 'Connecting',
  streaming: 'Streaming',
  reconnecting: 'Reconnecting',
  finishing: 'Finishing',
  done: 'Done',
  error: 'Error',
}

function EngineColumn({
  title,
  state,
  targetMs,
  rttMs,
  placeholder,
  showSplit,
}: {
  title: string | null
  state: TranscriptState
  targetMs: number
  rttMs: number | null
  placeholder: string
  showSplit: boolean
}) {
  const u = lastFinal(state.utterances)
  const lat = u ? utteranceLatency(u) : null
  const b = lat?.finalMs != null ? breakdown(lat.finalMs, u?.finalServerMs, rttMs) : null
  return (
    <div className="min-w-0">
      {title && (
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold">{title}</h3>
          <span className="text-[11px] text-muted-foreground">{state.backend ?? ''}</span>
        </div>
      )}
      <LatencyTimeline utterance={u} targetMs={targetMs} />
      <BudgetBar className="mt-4" measuredMs={lat?.finalMs ?? null} targetMs={targetMs} label="Speech end → final" />
      {showSplit && (
        <div className="mt-4 border-t border-border pt-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs text-muted-foreground">
            Where the wait went
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" className="rounded-sm" aria-label="About this split">
                  <Info className="size-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-64">
                Model time is what the recogniser reports for the final. Network is the measured round trip to the
                server. The rest is the silence the endpointer waits for before deciding you have stopped, plus
                queueing.
              </TooltipContent>
            </Tooltip>
          </div>
          <SegmentedBar
            ariaLabel="Latency split"
            segments={[
              {
                key: 'wait',
                label: 'Endpointing and queue',
                value: b?.waitMs ?? 0,
                color: 'var(--chart-mute)',
                display: b ? fmtMs(b.waitMs) : '–',
              },
              {
                key: 'model',
                label: 'Model',
                value: b?.modelMs ?? 0,
                color: 'var(--chart-1)',
                display: b ? fmtMs(b.modelMs) : '–',
              },
              {
                key: 'net',
                label: 'Network',
                value: b?.networkMs ?? 0,
                color: 'var(--chart-2)',
                display: b ? fmtMs(b.networkMs) : '–',
              },
            ]}
          />
        </div>
      )}
      <div className="sr-only" data-testid="engine-placeholder">
        {placeholder}
      </div>
    </div>
  )
}

function SessionStats({
  state,
  targetMs,
  title,
  compact,
}: {
  state: TranscriptState
  targetMs: number
  title?: string
  compact?: boolean
}) {
  const st = sessionStats(state)
  const v = (n: number) => (Number.isFinite(n) ? Math.round(n).toString() : '–')
  return (
    <div className="min-w-0">
      {title && <h3 className="mb-2 text-xs font-semibold">{title}</h3>}
      <div className={cn('grid grid-cols-2 gap-x-4 gap-y-3', !compact && 'sm:grid-cols-4')}>
        <Readout
          label="First partial p50"
          value={v(st.firstPartial.p50)}
          unit="ms"
          hint={`p95 ${v(st.firstPartial.p95)} ms`}
        />
        <Readout label="Final p50" value={v(st.final.p50)} unit="ms" hint={`p95 ${v(st.final.p95)} ms`} />
        <Readout
          label="Real-time factor"
          value={fmtRtf(state.rtf)}
          hint={state.rtf ? `${fmtSpeed(state.rtf)} real time` : 'lower is faster'}
        />
        <Readout label="Utterances" value={st.count} hint={`${fmtSeconds(state.audioSeconds)} of audio`} />
      </div>
      {st.count > 0 && <UtteranceList className="mt-4" utterances={state.utterances} targetMs={targetMs} />}
    </div>
  )
}

export default function SttPage() {
  const { models, model, select, isLoading: modelsLoading, error: modelsError } = useSelectedModel('stt')
  const p = useStore(prefs)
  const samples = useSamples()
  const catalog = useCatalog()
  const info = useInfo()
  const support = useSupport()
  const session = useSttSession()
  const [mode, setMode] = useState<EngineMode>(isDemo ? 'browser' : 'server')
  const [params, setParams] = useSearchParams()
  const shared = isDemo ? decodeShare(params.get('r')) : null
  const [telephony, setTelephony] = useState(false)
  const [liveDenoise, setLiveDenoise] = useState<Denoise>(defaultDenoise)
  const [dragging, setDragging] = useState(false)
  const [busySample, setBusySample] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const serverState = useTranscript(session.server)
  const browserState = useTranscript(session.browser)
  const running = session.phase !== 'idle'
  const httpRtt = usePing(mode !== 'browser')
  // Prefer the round trip measured on the stream itself; fall back to a small HTTP request.
  const rtt = serverState.rttMs ?? httpRtt

  const primary = mode === 'browser' ? browserState : serverState
  const [lastClip, setLastClip] = useState<Clip | null>(null)
  const speechRef = useRef(primary.speech)
  useEffect(() => {
    speechRef.current = primary.speech
  }, [primary.speech])

  // In-browser model and its language.
  const engine = sharedEngine()
  const engineState = useStore(engine.store)
  const bModel = browserModel(p.browserModel)
  const bLanguage = bModel.multilingual ? p.browserLanguage : (bModel.languages[0] ?? 'en')
  const browserLanguages = useMemo(() => [...new Set(BROWSER_MODELS.flatMap((m) => m.languages))], [])
  const [switchFor, setSwitchFor] = useState<string | null>(null)
  const [lidDecision, setLidDecision] = useState<LidDecision | null>(null)
  const [helpOpen, setHelpOpen] = useState(false)
  const lessData = useMemo(() => prefersLessData(), [])
  useEffect(() => {
    if (!running) engine.setModel(p.browserModel)
  }, [engine, p.browserModel, running])

  // Server models: the language list is everything the usable models cover; choosing one narrows the model list.
  const usableModels = useMemo(() => {
    const installed = models.filter((m) => m.installed)
    return installed.length ? installed : models
  }, [models])
  const languages = useMemo(() => {
    const set = new Set<string>()
    for (const m of usableModels) for (const l of m.languages) if (l !== 'multi') set.add(l)
    // A model listed only as "multi" takes any language: offer at least English and the Indian languages for it.
    if (set.size > 0 && usableModels.some((m) => m.languages.includes('multi'))) {
      set.add('en')
      for (const l of INDIC_LANGUAGES) if (l.code !== 'or') set.add(l.code)
    }
    return [...set]
  }, [usableModels])
  const language = languages.includes(p.language) ? p.language : 'auto'

  const chooseServerLanguage = (code: string) => {
    setPref('language', code)
    if (code === 'auto' || !model || supportsLanguage(model.languages, code)) return
    const other = usableModels.find((m) => m.installed && supportsLanguage(m.languages, code))
    if (other) {
      select(other.id)
      toast.message(`Switched to ${other.id}`, {
        description: `${model.id} does not support ${languageName(code)}.`,
      })
    }
  }
  const chooseBrowserLanguage = (code: string) => {
    if (code !== 'auto' && !bModel.languages.includes(code)) {
      setSwitchFor(code)
      return
    }
    if (bModel.multilingual) setPref('browserLanguage', code)
  }
  const chooseBrowserModel = (id: string) => {
    engine.setModel(id)
    pickBrowserModel(id)
  }

  const audioSamples = (samples.data?.samples ?? []).filter((s) => s.kind === 'audio' && s.url)
  const needsModel = mode !== 'browser'
  const canStart = !running && (!needsModel || Boolean(model?.installed))
  const micBlocked = !support.getUserMedia || !support.audioWorklet || !support.secureContext

  const begin = (input: SessionInput, browserLanguage: string = bLanguage) => {
    if (needsModel && !model) {
      toast.error('Choose a model first')
      return
    }
    setHelpOpen(false)
    session.start(input, {
      mode,
      model: model?.id ?? '',
      language,
      telephony: telephony && mode !== 'browser',
      denoise: mode !== 'browser' ? liveDenoise : undefined,
      browserLanguage,
    })
  }

  /**
   * The server's routing for `model=auto` with no language: the clip is heard by the language identifier first
   * and goes to a recogniser that speaks what it heard. Returns the language to tell the recogniser, if any.
   */
  const identify = async (samples: Float32Array, rate: number): Promise<LidDecision | null> => {
    const lid = p.browserLid
    if (!lid || mode !== 'browser') return null
    if (bModel.multilingual && bLanguage !== 'auto') return null
    try {
      const x = rate === STT_RATE ? samples : resample(samples, rate, STT_RATE)
      const heard = x.subarray(0, 30 * STT_RATE)
      const engine = audioEngine('lid')
      await engine.load(lid, p.browserAudioDevice)
      const { ranked, ms } = await engine.identify(heard)
      const decision = routeByLanguage(ranked, bModel, isDownloaded, { seconds: heard.length / STT_RATE, ms, lid })
      setLidDecision(decision)
      if (decision.reason === 'routed') chooseBrowserModel(decision.model)
      return decision
    } catch (err) {
      toast.error('Could not identify the language', { description: describeError(err) })
      return null
    }
  }

  const startFile = async (name: string, data: ArrayBuffer, browserLanguage?: string) => {
    // the Align panel starts from this recording (a copy: decoding may detach the buffer)
    setLastClip({ blob: new Blob([data.slice(0)]), name, seconds: null })
    try {
      const audio = await decodeAudio(data)
      if (audio.samples.length === 0) throw new Error('The file contains no audio')
      const decision = browserLanguage ? null : await identify(audio.samples, audio.sampleRate)
      begin(
        { kind: 'file', label: name, samples: audio.samples, sampleRate: audio.sampleRate },
        browserLanguage ?? decision?.language ?? undefined,
      )
    } catch (err) {
      toast.error(`Could not read ${name}`, { description: err instanceof Error ? err.message : undefined })
    }
  }

  const startSample = async (s: Sample) => {
    if (!s.url) return
    if (mode === 'browser' && p.browserLid) {
      // The language identifier decides, as for any file.
      setBusySample(s.id)
      try {
        await startFile(s.id, await api.bytes(s.url))
      } catch (err) {
        toast.error('Could not load the sample', { description: describeError(err) })
      } finally {
        setBusySample(null)
      }
      return
    }
    if (mode === 'browser' && s.language && !bModel.languages.includes(s.language)) {
      // The sample is in a language this model cannot do: offer the ones that can.
      setSwitchFor(s.language)
      return
    }
    let sampleLanguage: string | undefined
    if (mode === 'browser' && bModel.multilingual && s.language && bLanguage !== 'auto' && bLanguage !== s.language) {
      sampleLanguage = s.language
      setPref('browserLanguage', s.language)
      toast.message(`Language set to ${languageName(s.language)} for this sample`)
    }
    setBusySample(s.id)
    try {
      await startFile(s.id, await api.bytes(s.url), sampleLanguage)
    } catch (err) {
      toast.error('Could not load the sample', { description: describeError(err) })
    } finally {
      setBusySample(null)
    }
  }

  const onFiles = (files: FileList | null) => {
    const f = files?.[0]
    if (!f) return
    void f.arrayBuffer().then((buf) => startFile(f.name, buf))
  }

  // Microphone problems are explained in place (see MicHelpPanel); anything else is a toast.
  const micFailure: MicFailure | null =
    session.error?.mic ??
    (helpOpen ? { kind: 'denied', name: '', detail: '', elapsedMs: 0, permission: support.micPermission } : null)
  useEffect(() => {
    if (session.error && !session.error.mic) toast.error('Could not start', { description: session.error.message })
  }, [session.error])

  // Fetch the model before it is needed: when the demo is idle, and when a finger lands on the microphone button.
  const preload = useCallback(() => {
    void engine.load({ background: true }).catch(() => undefined)
  }, [engine])
  useEffect(() => {
    // Not after the page closed mid-load last time (most likely out of memory), and not on phones, where
    // memory is short and the visitor may be paying for data: there the model loads when it is asked for.
    if (!isDemo || mode !== 'browser' || !support.wasm || lessData || interrupted.get() || oneRuntimeAtATime()) return
    const w = globalThis as typeof globalThis & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number
      cancelIdleCallback?: (id: number) => void
    }
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(preload, { timeout: 3000 })
      return () => w.cancelIdleCallback?.(id)
    }
    const id = setTimeout(preload, 1200)
    return () => clearTimeout(id)
  }, [mode, support.wasm, lessData, preload, p.browserModel])
  const warmOnPress = mode !== 'server' && support.wasm ? preload : undefined
  useEffect(() => {
    if (serverState.status === 'error' && serverState.error)
      toast.error('Stream failed', { description: serverState.error })
  }, [serverState.status, serverState.error])

  const playTake = () => {
    const take = concatFloat(session.taken.current)
    if (take.length === 0) return
    const data = telephony ? simulateTelephony(take, STT_RATE) : take
    const rate = telephony ? TELEPHONY_RATE : STT_RATE
    const ctx = new AudioContext()
    const buf = ctx.createBuffer(1, data.length, rate)
    buf.getChannelData(0).set(data)
    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)
    src.onended = () => void ctx.close()
    src.start()
  }

  const copy = async () => {
    const text = transcriptText(primary)
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      toast.success('Transcript copied')
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }

  const hasTake = !running && session.input != null
  const loadText = mode !== 'server' && engineState.status === 'loading' ? 'Loading the model' : ''
  const statusLine =
    session.phase === 'idle'
      ? session.input
        ? `Last take: ${session.input.kind === 'mic' ? 'microphone' : session.input.label} · ${fmtSeconds(primary.audioSeconds)}`
        : 'Nothing recorded yet'
      : session.phase === 'starting' && session.mic === 'requesting'
        ? 'Waiting for microphone permission'
        : session.phase === 'starting' && session.mic === 'live'
          ? `Listening · ${loadText || STATUS_TEXT[primary.status]} · ${fmtSeconds(session.buffered ?? 0)} held`
          : session.phase === 'starting' && loadText
            ? `${loadText} · ${session.input?.label}`
            : `${STATUS_TEXT[primary.status]} · ${session.input?.kind === 'mic' ? 'microphone' : session.input?.label} · ${fmtSeconds(primary.audioSeconds)}`
  const playDefaultSample = () => {
    const wanted = bLanguage !== 'auto' && mode === 'browser' ? bLanguage : 'en'
    const s =
      audioSamples.find((x) => x.language === wanted && x.id !== 'en-fellow-americans') ??
      audioSamples.find((x) => x.id === 'en-ask-not') ??
      audioSamples[0]
    if (s) void startSample(s)
  }

  const stats = sessionStats(primary)
  const canShare = stats.count > 0 && Number.isFinite(stats.final.p50)
  const share = async () => {
    if (!canShare) return
    const url = shareUrl({
      m: bModel.label,
      d: (primary.backend ?? 'browser').replace(/^browser · /, ''),
      n: stats.count,
      fp: Number.isFinite(stats.firstPartial.p50) ? stats.firstPartial.p50 : null,
      f50: stats.final.p50,
      f95: stats.final.p95,
      rtf: primary.rtf,
    })
    try {
      if (navigator.share && /Mobi|Android/i.test(navigator.userAgent)) {
        await navigator.share({
          title: 'bolo live demo',
          text: `Speech recognised in my browser: final in ${Math.round(stats.final.p50)} ms`,
          url,
        })
      } else {
        await navigator.clipboard.writeText(url)
        toast.success('Link copied', { description: 'It carries the latency summary in the URL. Nothing is uploaded.' })
      }
    } catch {
      // cancelled share sheet, or clipboard blocked: show the link instead
      toast.message('Copy this link', { description: url })
    }
  }

  const toolbar = (
    <>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={mode}
        onValueChange={(v) => v && setMode(v as EngineMode)}
        aria-label="Where recognition runs"
        disabled={running}
      >
        <ToggleGroupItem
          value="server"
          className="px-3 text-xs"
          disabled={isDemo}
          title={isDemo ? SERVER_ONLY_HINT : undefined}
        >
          Server
        </ToggleGroupItem>
        <ToggleGroupItem value="browser" className="px-3 text-xs" disabled={!support.wasm} data-testid="mode-browser">
          In browser
        </ToggleGroupItem>
        <ToggleGroupItem
          value="compare"
          className="px-3 text-xs"
          disabled={!support.wasm || isDemo}
          title={isDemo ? SERVER_ONLY_HINT : undefined}
          data-testid="mode-compare"
        >
          Compare
        </ToggleGroupItem>
      </ToggleGroup>
      {mode === 'browser' ? (
        <>
          <BrowserModelPicker
            value={bModel.id}
            onChange={chooseBrowserModel}
            disabled={running}
            className="w-52 max-sm:min-w-44 max-sm:flex-1"
          />
          <LanguageSelect
            languages={browserLanguages}
            value={bLanguage}
            onChange={chooseBrowserLanguage}
            disabled={running}
            autoDetect={bModel.multilingual}
          />
        </>
      ) : (
        <>
          <ModelPicker
            models={models}
            value={model?.id ?? null}
            onChange={select}
            loading={modelsLoading}
            disabled={running}
            language={language}
            className="w-52 max-sm:min-w-44 max-sm:flex-1"
          />
          <LanguageSelect languages={languages} value={language} onChange={chooseServerLanguage} disabled={running} />
        </>
      )}
      <Tooltip>
        <TooltipTrigger asChild>
          <label
            className={cn(
              'flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30',
              (running || mode === 'browser') && 'cursor-not-allowed text-muted-foreground',
            )}
          >
            <Phone className="size-3.5 text-muted-foreground" aria-hidden />
            Phone line
            <Switch
              checked={telephony}
              onCheckedChange={setTelephony}
              disabled={running || mode === 'browser'}
              size="sm"
            />
          </label>
        </TooltipTrigger>
        <TooltipContent className="max-w-64">
          {isDemo && `${SERVER_ONLY_HINT} `}
          Downsamples to 8 kHz and applies G.711 μ-law in the browser, so the server hears what a phone call sounds
          like.
        </TooltipContent>
      </Tooltip>
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="ml-auto h-8 gap-1.5 font-normal">
            <Settings2 className="size-3.5" />
            Target <span className="num">{p.targetMs} ms</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-64">
          <Label htmlFor="target-ms" className="text-xs">
            Latency target (ms)
          </Label>
          <Input
            id="target-ms"
            type="number"
            min={50}
            max={5000}
            step={50}
            className="mt-1.5 h-8 num"
            value={p.targetMs}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (v >= 50 && v <= 5000) setPref('targetMs', Math.round(v))
            }}
          />
          <p className="mt-2 text-xs leading-4 text-muted-foreground">
            The longest acceptable gap between the end of speech and the final transcript. Used by the budget bars here
            and the target lines on the dashboard.
          </p>
        </PopoverContent>
      </Popover>
    </>
  )

  return (
    <Page title="Speech to text" toolbar={toolbar}>
      {mode !== 'server' && <InterruptedLoadNotice />}
      {micBlocked && !micFailure && (
        <Alert className="mb-4">
          <MicOff />
          <AlertTitle>Microphone capture is unavailable</AlertTitle>
          <AlertDescription>
            {!support.secureContext
              ? 'Browsers only allow microphone access on HTTPS or localhost. Open the console through one of those, or use a file or sample instead.'
              : 'This browser lacks AudioWorklet or getUserMedia (built-in browsers inside other apps often do). You can still upload a file or use a sample.'}
          </AlertDescription>
        </Alert>
      )}
      {modelsError && !modelsLoading && mode !== 'browser' && (
        <Alert className="mb-4" variant="destructive">
          <Info />
          <AlertTitle>Could not load the model list</AlertTitle>
          <AlertDescription>{describeError(modelsError)} In-browser mode still works.</AlertDescription>
        </Alert>
      )}
      {needsModel && model && isCloud(model) && model.installed && (
        <Alert className="mb-4" data-testid="cloud-model-notice">
          <Cloud />
          <AlertTitle>{cloudNotice(model)}</AlertTitle>
          <AlertDescription>
            <span>
              <span className="num">{model.id}</span> is served by the provider{' '}
              <span className="num">{model.provider?.name ?? 'unknown'}</span>: what you say here is sent there and
              billed by it.
            </span>
          </AlertDescription>
        </Alert>
      )}
      {needsModel && model && !model.installed && (
        <Alert className="mb-4">
          <Info />
          <AlertTitle>
            <span className="num">{model.id}</span> is not installed
          </AlertTitle>
          <AlertDescription>{notInstalledHint(info.data, model)}</AlertDescription>
        </Alert>
      )}

      {shared && (
        <div
          className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-border bg-card px-4 py-3"
          data-testid="shared-result"
        >
          <div className="min-w-0">
            <div className="text-xs text-muted-foreground">
              Shared result · measured on someone else&apos;s device, not yours
            </div>
            <div className="mt-0.5 truncate num text-[13px]">
              {shared.m} · {shared.d}
            </div>
          </div>
          <Readout
            label="Final p50"
            value={Math.round(shared.f50)}
            unit="ms"
            hint={`p95 ${Math.round(shared.f95)} ms`}
          />
          {shared.fp != null && <Readout label="First partial p50" value={Math.round(shared.fp)} unit="ms" />}
          {shared.rtf != null && (
            <Readout label="Real-time factor" value={fmtRtf(shared.rtf)} hint={`${fmtSpeed(shared.rtf)} real time`} />
          )}
          <Readout label="Utterances" value={shared.n} />
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={() => {
              params.delete('r')
              setParams(params, { replace: true })
            }}
          >
            Dismiss
          </Button>
        </div>
      )}
      {isDemo && !running && session.input == null && (
        <div className="mb-4 rounded-xl border border-border bg-card px-5 py-6 sm:px-8 sm:py-8" data-testid="demo-hero">
          <h2 className="text-2xl font-semibold tracking-tight text-balance sm:text-3xl">
            Try it: press the mic and talk.
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
            A speech recogniser will run right here in this tab and show you where every millisecond went. Nothing you
            say leaves your device. The first run downloads the {bModel.label} model ({fmtBytes(bModel.approxBytes)},{' '}
            {bModel.license}) plus a 14 MB runtime; after that both are cached and it works offline.
          </p>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Button
              className="h-10 px-4 text-[15px]"
              onPointerDown={lessData ? undefined : warmOnPress}
              onClick={() => begin({ kind: 'mic', label: 'microphone' })}
              disabled={micBlocked || !support.wasm}
              data-testid="hero-mic"
            >
              <Mic /> Start talking
            </Button>
            <Button
              variant="outline"
              className="h-10 px-4 text-[15px]"
              onClick={playDefaultSample}
              disabled={!support.wasm || audioSamples.length === 0 || busySample != null}
              data-testid="hero-sample"
            >
              <Play className="fill-current" /> No microphone? Play a sample
            </Button>
            {!bModel.languages.includes('hi') && (
              <Button
                variant="ghost"
                className="h-10 px-3 text-[15px]"
                onClick={() => chooseBrowserLanguage('hi')}
                disabled={!support.wasm}
                data-testid="hero-hindi"
              >
                <span lang="hi">हिन्दी</span> · Try Hindi
              </Button>
            )}
          </div>
          <div className="mt-4 flex max-w-2xl flex-col gap-2">
            {lessData && engineState.status === 'idle' && !engineState.cached && (
              <p className="text-xs leading-5 text-muted-foreground" data-testid="data-saver-note">
                Your browser asks sites to save data, so the model is not fetched in advance. It downloads (
                {fmtBytes(bModel.approxBytes)}) when you press a button above or the model status.
              </p>
            )}
            <MicStatusLine
              permission={support.micPermission}
              available={!micBlocked}
              onShowHelp={() => setHelpOpen(true)}
            />
          </div>
          {!support.wasm && (
            <p className="mt-3 text-sm text-warning-text">
              This browser does not support WebAssembly, which the demo needs.
            </p>
          )}
        </div>
      )}

      {micFailure && (
        <MicHelpPanel
          className="mb-4"
          failure={micFailure}
          onRetry={() => begin({ kind: 'mic', label: 'microphone' })}
          onSample={audioSamples.length > 0 && canStart ? playDefaultSample : undefined}
          onUpload={canStart ? () => fileInput.current?.click() : undefined}
          onDismiss={() => {
            setHelpOpen(false)
            session.clearError()
          }}
        />
      )}
      <SwitchModelDialog
        language={switchFor}
        current={bModel}
        onCancel={() => setSwitchFor(null)}
        onChoose={(id) => {
          chooseBrowserModel(id)
          if (switchFor) setPref('browserLanguage', switchFor)
          setSwitchFor(null)
        }}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-4 xl:col-start-1 xl:row-start-1">
          <Panel
            title="Input"
            description={statusLine}
            actions={
              hasTake ? (
                <Button size="sm" variant="ghost" onClick={playTake}>
                  <Volume2 /> {telephony ? 'Hear it as a phone line' : 'Replay'}
                </Button>
              ) : undefined
            }
            onDragOver={(e) => {
              e.preventDefault()
              if (!running) setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              if (!running) onFiles(e.dataTransfer.files)
            }}
            className={cn('transition-colors duration-150', dragging && 'border-ring bg-chart-1/5')}
          >
            <div className="flex flex-wrap items-center gap-2">
              {running ? (
                <Button
                  onClick={() => void session.stop()}
                  variant="destructive"
                  className="h-9 min-w-28"
                  disabled={session.phase === 'finishing'}
                  data-testid="stop"
                >
                  <Square className="fill-current" /> {session.phase === 'finishing' ? 'Finishing…' : 'Stop'}
                </Button>
              ) : (
                <Button
                  onPointerDown={lessData ? undefined : warmOnPress}
                  onClick={() => begin({ kind: 'mic', label: 'microphone' })}
                  className="h-9 min-w-28"
                  disabled={!canStart || micBlocked}
                  data-testid="mic"
                >
                  <Mic /> Start talking
                </Button>
              )}
              <Button variant="outline" className="h-9" onClick={() => fileInput.current?.click()} disabled={!canStart}>
                <Upload /> Upload audio
              </Button>
              <input
                ref={fileInput}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
                className="sr-only"
                tabIndex={-1}
                aria-label="Audio file"
                onChange={(e) => {
                  onFiles(e.target.files)
                  e.target.value = ''
                }}
              />
              <span className="hidden text-xs text-muted-foreground sm:inline">or drop a file here</span>
              {mode !== 'server' && <ModelStatusChip className="sm:ml-auto" />}
            </div>

            {session.phase === 'starting' && (
              <p
                role="status"
                className="mt-3 rounded-lg border border-chart-1/30 bg-chart-1/10 px-3 py-2 text-[13px] leading-5"
                data-testid="starting-note"
                data-mic={session.mic}
              >
                {session.mic === 'requesting' ? (
                  'Your browser is asking for the microphone. Choose Allow to start.'
                ) : session.mic === 'live' ? (
                  <>
                    <span className="font-medium">The microphone is on. Go ahead and talk.</span>{' '}
                    {mode === 'server' ? 'Connecting to the server.' : 'The model is still getting ready.'} What you say
                    is being kept (
                    <span className="num" data-testid="buffered-seconds">
                      {(session.buffered ?? 0).toFixed(1)} s
                    </span>{' '}
                    so far) and will be transcribed the moment it is.
                  </>
                ) : mode === 'server' ? (
                  'Connecting to the server.'
                ) : (
                  'The audio starts as soon as the model is ready.'
                )}
              </p>
            )}
            {mode !== 'server' && (
              <LoadTrouble className="mt-3" onSample={audioSamples.length > 0 ? playDefaultSample : undefined} />
            )}

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-xs text-muted-foreground">Samples</span>
              {samples.isLoading && (
                <>
                  <Skeleton className="h-7 w-28" />
                  <Skeleton className="h-7 w-28" />
                </>
              )}
              {samples.isError && <span className="text-xs text-muted-foreground">unavailable</span>}
              {!samples.isLoading && !samples.isError && audioSamples.length === 0 && (
                <span className="text-xs text-muted-foreground">none on this server</span>
              )}
              {audioSamples.map((s) => (
                <Tooltip key={s.id}>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={!canStart || busySample != null}
                      onClick={() => void startSample(s)}
                      data-testid={`sample-${s.id}`}
                    >
                      {busySample === s.id ? <FileAudio className="live-dot" /> : <Play className="fill-current" />}
                      <span className="num text-xs">{s.id}</span>
                      <span className="num text-muted-foreground">
                        {s.seconds != null ? `${s.seconds.toFixed(1)}s` : ''}
                      </span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-72">
                    {languageName(s.language)} · “{s.text}”
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>

            {mode !== 'browser' && !isDemo && (
              <div className="mt-3 grid gap-1">
                <DenoiseControl live value={liveDenoise} onChange={setLiveDenoise} disabled={running} />
                <LiveDenoiseLine state={serverState} />
              </div>
            )}

            <div className="mt-4 rounded-lg border border-border bg-background/60 px-2 py-1">
              <LevelStrip
                track={session.levels}
                speech={speechRef}
                active={running}
                label={running ? 'Audio level, live' : 'Audio level of the last take'}
              />
            </div>
            <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
              <span className="flex items-center gap-3">
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-3 rounded-[2px] bg-chart-1" aria-hidden /> speech
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-3 rounded-[2px] bg-chart-mute" aria-hidden /> silence
                </span>
              </span>
              <span className="num">last 12 s</span>
            </div>
          </Panel>

          <Panel
            title="Transcript"
            description={
              primary.finals.length
                ? `${primary.finals.length} final segment${primary.finals.length === 1 ? '' : 's'}`
                : 'Solid text is final; lighter text may still change'
            }
            actions={
              <Button size="sm" variant="ghost" onClick={() => void copy()} disabled={primary.finals.length === 0}>
                <Copy /> Copy
              </Button>
            }
          >
            {mode === 'compare' ? (
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="min-w-0">
                  <div className="mb-1 eyebrow">Server · {model?.id ?? '–'}</div>
                  <Transcript state={serverState} placeholder="The server transcript appears here." />
                </div>
                <div className="min-w-0 sm:border-l sm:border-border sm:pl-4">
                  <div className="mb-1 eyebrow">In browser · {bModel.id}</div>
                  <Transcript state={browserState} placeholder="The in-browser transcript appears here." />
                </div>
              </div>
            ) : (
              <Transcript
                state={primary}
                placeholder="Press Start talking, or play a sample. Words appear as you speak."
              />
            )}
          </Panel>
          <Panel
            title="This session"
            description="Measured in this browser, from audio leaving the page to text arriving"
            actions={
              isDemo ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void share()}
                  disabled={!canShare}
                  data-testid="share"
                >
                  <Share2 /> Share this result
                </Button>
              ) : undefined
            }
          >
            {mode === 'compare' ? (
              <div className="grid gap-6 lg:grid-cols-2">
                <SessionStats state={serverState} targetMs={p.targetMs} title="Server" compact />
                <SessionStats state={browserState} targetMs={p.targetMs} title="In browser" compact />
              </div>
            ) : (
              <SessionStats state={primary} targetMs={p.targetMs} />
            )}
          </Panel>
        </div>

        <div className="flex min-w-0 flex-col gap-4 xl:col-start-2 xl:row-start-1">
          <Panel
            title="Latency timeline"
            description="Last utterance"
            actions={
              mode !== 'browser' ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span
                      tabIndex={0}
                      className="rounded-md border border-border px-1.5 py-0.5 num text-[11px] text-muted-foreground"
                      data-testid="rtt"
                    >
                      network {rtt != null ? `${Math.max(1, Math.round(rtt))} ms` : '–'}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Median round trip to the server: a ping on the open stream, or a small HTTP request when idle.
                  </TooltipContent>
                </Tooltip>
              ) : undefined
            }
          >
            {mode === 'compare' ? (
              <div className="grid gap-6">
                <EngineColumn
                  title="Server"
                  state={serverState}
                  targetMs={p.targetMs}
                  rttMs={rtt}
                  placeholder="server"
                  showSplit
                />
                <div className="border-t border-border pt-5">
                  <EngineColumn
                    title="In browser"
                    state={browserState}
                    targetMs={p.targetMs}
                    rttMs={0}
                    placeholder="browser"
                    showSplit
                  />
                </div>
              </div>
            ) : (
              <EngineColumn
                title={null}
                state={primary}
                targetMs={p.targetMs}
                rttMs={mode === 'browser' ? 0 : rtt}
                placeholder={mode}
                showSplit
              />
            )}
          </Panel>

          {mode !== 'server' && (
            <BrowserProcessingPanel
              disabled={running}
              removed={browserState.repeatsRemoved}
              denoiser={session.denoiser}
              decision={lidDecision}
              onSuggest={(lang) => setSwitchFor(lang)}
              recogniserWritesNone={Boolean(bModel.noPunctuation)}
            />
          )}
          {mode !== 'server' && <DownloadsPanel />}
          {mode !== 'server' && <DevicePanel />}
          {mode !== 'server' && (
            <BrowserEnginePanel
              support={support}
              language={bLanguage}
              picker={
                mode === 'compare' ? (
                  <div className="mb-3 flex flex-wrap gap-2">
                    <BrowserModelPicker
                      value={bModel.id}
                      onChange={chooseBrowserModel}
                      disabled={running}
                      className="min-w-44 flex-1"
                    />
                    <LanguageSelect
                      languages={browserLanguages}
                      value={bLanguage}
                      onChange={chooseBrowserLanguage}
                      disabled={running}
                      autoDetect={bModel.multilingual}
                    />
                  </div>
                ) : undefined
              }
            />
          )}

          {mode === 'server' && !isDemo && <LanguageIdPanel info={info.data} samples={samples.data?.samples ?? []} />}
          {mode === 'server' && !isDemo && (
            <AlignPanel
              info={info.data}
              samples={samples.data?.samples ?? []}
              initialClip={running ? null : lastClip}
              initialText={running ? '' : transcriptText(primary)}
            />
          )}
          {mode === 'server' && (
            <Panel title="Try it without the network" bodyClassName="flex items-start gap-3">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/50 text-muted-foreground">
                <Mic className="size-4" aria-hidden />
              </span>
              <div className="min-w-0 text-[13px] leading-5 text-muted-foreground">
                <p>
                  <span className="text-foreground">In browser</span> runs a small model on this device with WebAssembly
                  (English, or multilingual Whisper for Hindi and other languages): no round trip at all.{' '}
                  <span className="text-foreground">Compare</span> feeds the same audio to both and lines the timelines
                  up.
                </p>
                {!support.wasm && <p className="mt-1 text-warning-text">This browser does not support WebAssembly.</p>}
              </div>
            </Panel>
          )}
        </div>
      </div>
      {!isDemo && mode === 'server' && (
        <SpeakersPanel
          className="mt-4"
          info={info.data}
          catalog={catalog.data?.models ?? []}
          model={model}
          language={language}
          samples={samples.data?.samples ?? []}
        />
      )}
    </Page>
  )
}

/** What the server said about denoising the live stream: the model that runs (or why it was skipped) and its cost. */
function LiveDenoiseLine({ state }: { state: TranscriptState }) {
  const d = state.denoise
  const m = state.denoiseMetrics
  if (d === undefined || state.status === 'idle') return null
  return (
    <p
      className={cn('text-xs leading-4', d?.status === 'skipped' ? 'text-warning-text' : 'text-muted-foreground')}
      data-testid="live-denoise"
    >
      Server:{' '}
      {d == null ? (
        'no denoising'
      ) : d.status === 'skipped' ? (
        <>
          <span className="num">{d.model}</span> skipped{d.detail ? `: ${d.detail}` : ''}
        </>
      ) : (
        <>
          denoising with <span className="num">{d.model}</span> at strength {d.strength}
          {d.stateful ? ', keeping its state from chunk to chunk' : ', each chunk with the half second before it'}
          {m?.chunks ? (
            <span className="num">
              {' '}
              · {m.mean_ms?.toFixed(1)} ms per chunk (p95 {m.p95_ms?.toFixed(1)}) · {Math.round(m.held_ms ?? 0)} ms held
              back
            </span>
          ) : null}
        </>
      )}
    </p>
  )
}
