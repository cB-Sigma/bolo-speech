import {
  CircleCheck,
  Cloud,
  Download,
  Info,
  Languages,
  Play,
  Repeat2,
  Square,
  Tag,
  TriangleAlert,
  Volume2,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { BudgetBar } from '@/components/charts/budget-bar'
import { Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { Empty, Panel, Readout } from '@/components/panel'
import { ChunkTimeline } from '@/components/tts/chunk-timeline'
import { PlaybackWave } from '@/components/tts/playback-wave'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ComponentNotice } from '@/components/component-install'
import {
  describeError,
  missingComponent,
  useComponents,
  useInfo,
  useSamples,
  useSelectedModel,
  useWordLists,
} from '@/hooks/use-api'
import { WordListSelect } from '@/components/wordlists/wordlist-select'
import { Switch } from '@/components/ui/switch'
import { useStore } from '@/hooks/use-store'
import type { Sample } from '@/lib/api/types'
import { STT_RATE } from '@/lib/audio/capture'
import { resample } from '@/lib/audio/resample'
import { encodeWav } from '@/lib/audio/wav'
import { download } from '@/lib/bench'
import { fmtMs, fmtRtf, fmtSeconds, fmtSpeed, languageName } from '@/lib/format'
import { wordErrorRate } from '@/lib/latency/wer'
import { prefs, setPref } from '@/lib/prefs'
import { Store } from '@/lib/store'
import { ServerTranscriber } from '@/lib/stt/server'
import { transcriptText } from '@/lib/stt/transcriber'
import { emptyTts, TtsSession, type TtsState, type TtsTransport } from '@/lib/tts/session'
import { RoundTripPanel, type RoundTrip } from '@/components/tts/round-trip'
import { SpeechCaptions } from '@/components/tts/speech-captions'
import { isDemo } from '@/lib/api/config'
import { detectSupport } from '@/lib/support'
import { BrowserTts } from './tts-browser'
import { VoiceClonePanel, type CloneReference } from '@/components/tts/voice-clone'
import { startingBlend, VoiceBlendPanel } from '@/components/tts/voice-blend'
import {
  blendString,
  canBlend,
  languageSummary,
  languageWarning,
  parseLanguageHeader,
  type BlendPart,
} from '@/lib/blend'
import { dataUrlToBlob, type VoiceSource } from '@/components/voices/new-voice-dialog'
import { useVoices } from '@/components/voices/use-voices'
import { VoiceLibrary } from '@/components/voices/voice-library'
import {
  canClone,
  followsInstructions,
  cloudNotice,
  isCloud,
  notInstalledHint,
  PHONEMIZER_LABEL,
  phonemizerChoices,
  voiceFamily,
} from '@/lib/capabilities'
import {
  isIndicLanguage,
  ROMANISED_HINT,
  ROMANISED_LABEL,
  ROMANISED_MODES,
  romanisedBlocked,
  romanisedOffered,
  type RomanisedMode,
} from '@/lib/romanised'
import { cn } from '@/lib/utils'

const FALLBACK_TEXTS: Sample[] = [
  {
    id: 'en',
    kind: 'text',
    language: 'en',
    text: 'Thanks for calling. I can help you check an order, change a delivery slot, or talk to a person. What would you like to do?',
  },
  { id: 'hi', kind: 'text', language: 'hi', text: 'नमस्ते, बोलो में आपका स्वागत है। मैं आपकी क्या मदद कर सकता हूँ?' },
  { id: 'es', kind: 'text', language: 'es', text: 'Hola, gracias por llamar. ¿En qué puedo ayudarte hoy?' },
]

const idle = new Store<TtsState>(emptyTts())
/** The voice choice that names no voice: the server picks one for the language it detects. */
const AUTO_VOICE = 'auto'

export default function TtsPage() {
  const [mode, setMode] = useState<'server' | 'browser'>(isDemo ? 'browser' : 'server')
  const support = detectSupport()
  const toggle = isDemo ? null : (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={mode}
      onValueChange={(v) => v && setMode(v as 'server' | 'browser')}
      aria-label="Where synthesis runs"
      data-testid="tts-mode"
    >
      <ToggleGroupItem value="server" className="px-3 text-xs" data-testid="tts-mode-server">
        Server
      </ToggleGroupItem>
      <ToggleGroupItem value="browser" className="px-3 text-xs" disabled={!support.wasm} data-testid="tts-mode-browser">
        In browser
      </ToggleGroupItem>
    </ToggleGroup>
  )
  return mode === 'browser' ? <BrowserTts modeToggle={toggle} /> : <ServerTts modeToggle={toggle} />
}

function ServerTts({ modeToggle }: { modeToggle: ReactNode }) {
  const info = useInfo()
  const { models, model, select, isLoading, error: modelsError } = useSelectedModel('tts')
  const stt = useSelectedModel('stt')
  const samples = useSamples()
  const p = useStore(prefs)
  const [text, setText] = useState(FALLBACK_TEXTS[0]?.text ?? '')
  const [transport, setTransport] = useState<TtsTransport>('websocket')
  const [session, setSession] = useState<TtsSession | null>(null)
  const [trip, setTrip] = useState<RoundTrip | null>(null)
  const previewRef = useRef<TtsSession | null>(null)
  const state = useStore(session?.store ?? idle)
  const busy = state.status === 'connecting' || state.status === 'synthesizing' || state.status === 'playing'

  const voices = useMemo(() => model?.voices ?? [], [model])
  const voice = voices.find((v) => v.id === p.ttsVoice) ?? voices[0] ?? null
  // Per-request options. Empty means "the voice's own" (or what is pinned for it on the server).
  const [language, setLanguage] = useState('')
  const [phonemizer, setPhonemizer] = useState('')
  const [cloning, setCloning] = useState(false)
  const [reference, setReference] = useState<CloneReference | null>(null)
  const [ssml, setSsml] = useState(false)
  // word captions of the speech (forced alignment on the server), and the text they caption
  const [captions, setCaptions] = useState(false)
  const [spoken, setSpoken] = useState('')
  const [lists, setLists] = useState<string[]>([])
  const [romanised, setRomanised] = useState('')
  const [instructions, setInstructions] = useState('')
  const wordLists = useWordLists()
  // Saved voices (the server's library): one can stand in for the model's voice when the model clones voices.
  const library = useVoices()
  const [savedId, setSavedId] = useState<string | null>(null)
  // Blends of the model's own voices (Kokoro): built here, or saved in the library.
  const [blending, setBlending] = useState(false)
  const [parts, setParts] = useState<BlendPart[]>([])
  const [blendId, setBlendId] = useState<string | null>(null)
  const [detect, setDetect] = useState(true)
  const [toSave, setToSave] = useState<VoiceSource | null>(null)
  const listNames = (wordLists.data?.wordlists ?? []).map((l) => l.name)
  const chosenLists = lists.filter((n) => listNames.includes(n))
  const languages = useMemo(() => (model?.languages ?? []).filter((l) => l !== 'multi'), [model])
  const multilingual = languages.length > 1
  const lang = multilingual && languages.includes(language) ? language : ''
  const phonemizers = phonemizerChoices(model, voice)
  const phon =
    phonemizers.length && (phonemizer === 'auto' || phonemizers.some((c) => c.id === phonemizer)) ? phonemizer : ''
  const pinned = typeof model?.settings?.phonemizer === 'string' ? model.settings.phonemizer : null
  // Romanised text: offered for voices of Indian languages by a server that announces the option.
  const voiceLanguage = lang || voice?.language || languages[0] || ''
  const romanisedCap = isIndicLanguage(voiceLanguage) ? romanisedOffered(info.data) : null
  const romanisedPin = typeof model?.settings?.romanised === 'string' ? model.settings.romanised : null
  const romanisedDefault = romanisedPin ?? romanisedCap?.default ?? 'off'
  const romanisedNote = romanisedCap ? romanisedBlocked(info.data, voiceLanguage) : null
  const roman = romanisedCap && ROMANISED_MODES.includes(romanised as RomanisedMode) ? romanised : ''
  const saved = canClone(model) && savedId ? (library.data?.voices.find((v) => v.id === savedId) ?? null) : null
  const clone = canClone(model) && cloning && !saved
  const blends = canBlend(model)
  // A blend names voices of the model: parts naming voices it lacks start over from its first two.
  const blendParts =
    parts.length && parts.every((x) => voices.some((v) => v.id === x.voice)) ? parts : startingBlend(voices)
  const blendSpelled = blends && blending && !clone ? blendString(blendParts) : ''
  const savedBlend =
    blends && blendId && !blendSpelled ? (library.data?.blends?.find((b) => b.id === blendId) ?? null) : null
  // "From the text": no voice is named, so the server picks one that speaks the language it detects.
  const autoVoice = multilingual && p.ttsVoice === AUTO_VOICE && detect && !lang
  const voiceSent = blendSpelled || (savedBlend ? savedBlend.id : saved ? saved.id : autoVoice ? undefined : voice?.id)
  const cloneReady = !clone || reference != null
  const steerable = followsInstructions(model)
  const direction = steerable ? instructions.trim() : ''
  const requestOptions = {
    language: lang || undefined,
    phonemizer: phon || undefined,
    referenceAudio: clone ? reference?.dataUrl : undefined,
    referenceText: clone ? reference?.text || undefined : undefined,
    inputFormat: ssml ? ('ssml' as const) : undefined,
    wordlists: chosenLists.length ? chosenLists : undefined,
    romanised: roman || undefined,
    detectLanguage: detect ? undefined : false,
    instructions: direction || undefined,
  }
  const forcedHttp = clone || ssml
  const fromServer = (samples.data?.samples ?? []).filter((s) => s.kind === 'text')
  const texts = fromServer.length ? fromServer : FALLBACK_TEXTS

  useEffect(() => () => void session?.dispose(), [session])
  useEffect(() => () => void previewRef.current?.dispose(), [])
  useEffect(() => {
    // A missing component gets an inline notice with a way forward instead of a toast.
    if (state.status === 'error' && state.error && state.errorCode !== 'component_missing') {
      toast.error('Synthesis failed', { description: state.error })
    }
  }, [state.status, state.error, state.errorCode])

  const speak = (): TtsSession | null => {
    if (!model || !text.trim()) return null
    void previewRef.current?.dispose()
    const s = new TtsSession({
      model: model.id,
      voice: voiceSent,
      text: text.trim(),
      transport,
      ...requestOptions,
    })
    setSession(s)
    setSpoken(ssml ? '' : text.trim())
    setTrip(null)
    s.start()
    return s
  }

  const preview = () => {
    if (!model || !voice) return
    void previewRef.current?.dispose()
    session?.stop()
    const s = new TtsSession({
      model: model.id,
      voice: voice.id,
      text: voice.language.startsWith('hi') ? `नमस्ते, मैं ${voice.name} हूँ।` : `Hello, this is ${voice.name}.`,
      language: lang || undefined,
      phonemizer: phon || undefined,
      wordlists: chosenLists.length ? chosenLists : undefined,
    })
    previewRef.current = s
    s.start()
  }

  const save = () => {
    if (!session || !state.sampleRate) return
    download(`bolo-${model?.id ?? 'speech'}.wav`, encodeWav(session.audio(), state.sampleRate), 'audio/wav')
  }

  /** Feed the synthesised audio to the recogniser and score the transcript against the input text. */
  const transcribe = async (s: TtsSession, reference: string) => {
    const sttModel = stt.model
    if (!sttModel?.installed) {
      toast.error('No speech-to-text model is installed for the round trip')
      return
    }
    setTrip({ status: 'running', model: sttModel.id, hypothesis: '', result: null })
    try {
      // Wait until synthesis (not playback) has finished.
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
      const st = s.store.get()
      const audio = resample(s.audio(), st.sampleRate ?? 24000, STT_RATE)
      const t = new ServerTranscriber({ model: sttModel.id })
      await t.start()
      const block = STT_RATE / 50
      for (let o = 0; o < audio.length; o += block) t.push(audio.subarray(o, Math.min(audio.length, o + block)))
      t.push(new Float32Array(STT_RATE * 0.6)) // trailing silence so the endpointer closes the last utterance
      await t.finish()
      const final = t.store.get()
      if (final.status === 'error') throw new Error(final.error ?? 'Transcription failed')
      const hypothesis = transcriptText(final)
      setTrip({ status: 'done', model: sttModel.id, hypothesis, result: wordErrorRate(reference, hypothesis) })
    } catch (err) {
      setTrip({ status: 'error', model: sttModel.id, hypothesis: '', result: null, error: describeError(err) })
    }
  }

  const roundTrip = () => {
    const s = state.totalMs != null && session && !busy ? session : speak()
    if (s) void transcribe(s, text.trim())
  }

  const hasAudio = state.audioSeconds > 0 && state.totalMs != null
  const savedChoices = canClone(model) ? (library.data?.voices ?? []) : []
  const blendChoices = blends ? (library.data?.blends ?? []) : []
  const reported = state.reported
  const decision = parseLanguageHeader(reported?.language)
  const decisionWarning = languageWarning(decision)
  const components = useComponents().data?.components
  // Known up front from the catalog, or learnt the hard way from a `component_missing` error.
  const missing =
    missingComponent(model, components) ??
    (state.errorCode === 'component_missing' ? (components?.find((c) => !c.installed) ?? null) : null)
  const blocked = missing != null && model?.available === false
  const underrun = state.underruns > 0

  const toolbar = (
    <>
      <ModelPicker
        models={models}
        value={model?.id ?? null}
        onChange={select}
        loading={isLoading}
        disabled={busy}
        className="w-52 max-sm:min-w-44 max-sm:flex-1"
        groupBy={voiceFamily}
      />
      <div className="flex items-center gap-1">
        <Select
          value={
            savedBlend
              ? `blend:${savedBlend.id}`
              : saved
                ? `saved:${saved.id}`
                : autoVoice
                  ? AUTO_VOICE
                  : (voice?.id ?? undefined)
          }
          onValueChange={(v) => {
            if (v.startsWith('saved:')) {
              setSavedId(v.slice(6))
              setBlendId(null)
              return
            }
            if (v.startsWith('blend:')) {
              setBlendId(v.slice(6))
              setSavedId(null)
              return
            }
            setSavedId(null)
            setBlendId(null)
            setPref('ttsVoice', v)
          }}
          disabled={busy || (voices.length === 0 && !savedChoices.length) || clone || Boolean(blendSpelled)}
        >
          <SelectTrigger className="h-8 w-48" aria-label="Voice" data-testid="voice-picker">
            <SelectValue placeholder={voices.length ? 'Voice' : 'Default voice'} />
          </SelectTrigger>
          <SelectContent position="popper" align="start" className="max-h-72">
            {multilingual && (
              <SelectItem value={AUTO_VOICE} disabled={!detect || Boolean(lang)}>
                From the text
                <span className="ml-2 text-xs text-muted-foreground">a voice of the detected language</span>
              </SelectItem>
            )}
            {voices.map((v) => (
              <SelectItem key={v.id} value={v.id}>
                {v.name}
                <span className="ml-2 num text-xs text-muted-foreground">
                  {v.language}
                  {v.gender ? ` · ${v.gender}` : ''}
                </span>
              </SelectItem>
            ))}
            {savedChoices.length > 0 && (
              <SelectGroup>
                {voices.length > 0 && <SelectSeparator />}
                <SelectLabel className="text-xs">Saved voices (cloned)</SelectLabel>
                {savedChoices.map((v) => (
                  <SelectItem key={v.id} value={`saved:${v.id}`}>
                    {v.name}
                    <span className="ml-2 num text-xs text-muted-foreground">{v.language ?? 'saved'}</span>
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
            {blendChoices.length > 0 && (
              <SelectGroup>
                {(voices.length > 0 || savedChoices.length > 0) && <SelectSeparator />}
                <SelectLabel className="text-xs">Saved blends</SelectLabel>
                {blendChoices.map((b) => (
                  <SelectItem key={b.id} value={`blend:${b.id}`}>
                    {b.name}
                    <span className="ml-2 num text-xs text-muted-foreground">blend</span>
                  </SelectItem>
                ))}
              </SelectGroup>
            )}
          </SelectContent>
        </Select>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              onClick={preview}
              disabled={
                !voice || busy || model?.available === false || saved != null || savedBlend != null || !!blendSpelled
              }
              aria-label="Preview this voice"
            >
              <Volume2 />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Preview this voice</TooltipContent>
        </Tooltip>
      </div>
      {multilingual && (
        <Select value={lang || 'voice'} onValueChange={(v) => setLanguage(v === 'voice' ? '' : v)} disabled={busy}>
          <SelectTrigger className="h-8 w-40" aria-label="Language" data-testid="tts-language">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" align="start" className="max-h-72">
            <SelectItem value="voice">Language: auto</SelectItem>
            {languages.map((l) => (
              <SelectItem key={l} value={l}>
                {languageName(l)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {phonemizers.length > 0 && (
        <Select
          value={phon || 'default'}
          onValueChange={(v) => setPhonemizer(v === 'default' ? '' : v)}
          disabled={busy}
        >
          <SelectTrigger className="h-8 w-48" aria-label="Phonemiser" data-testid="phonemizer">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" align="start" className="max-w-80">
            <SelectItem value="default">
              Phonemes: {pinned ? `pinned (${PHONEMIZER_LABEL[pinned] ?? pinned})` : 'default'}
            </SelectItem>
            <SelectItem value="auto">Phonemes: automatic</SelectItem>
            {phonemizers.map((c) => (
              <SelectItem key={c.id} value={c.id} disabled={!c.available}>
                <span className="flex flex-col">
                  <span>
                    {PHONEMIZER_LABEL[c.id] ?? c.id}
                    {c.default ? ' (default)' : ''}
                  </span>
                  {c.note && <span className="text-xs text-muted-foreground">{c.note}</span>}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {modeToggle && <div className="ml-auto">{modeToggle}</div>}
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={forcedHttp ? 'http' : transport}
        onValueChange={(v) => v && setTransport(v as TtsTransport)}
        aria-label="Transport"
        disabled={busy || forcedHttp}
        className={modeToggle ? undefined : 'ml-auto'}
      >
        <ToggleGroupItem value="websocket" className="px-3 text-xs">
          WebSocket
        </ToggleGroupItem>
        <ToggleGroupItem value="http" className="px-3 text-xs">
          Chunked HTTP
        </ToggleGroupItem>
      </ToggleGroup>
    </>
  )

  return (
    <Page title="Text to speech" toolbar={toolbar}>
      {modelsError && !isLoading && (
        <Alert className="mb-4" variant="destructive">
          <Info />
          <AlertTitle>Could not load the model list</AlertTitle>
          <AlertDescription>{describeError(modelsError)}</AlertDescription>
        </Alert>
      )}
      {!isLoading && !modelsError && models.length === 0 && (
        <Alert className="mb-4">
          <Info />
          <AlertTitle>No text-to-speech models</AlertTitle>
          <AlertDescription>Pull one from the Models screen to use this playground.</AlertDescription>
        </Alert>
      )}
      {missing && model && <ComponentNotice component={missing} subject={model.id} className="mb-4" />}
      {model && isCloud(model) && model.installed && (
        <Alert className="mb-4" data-testid="cloud-model-notice">
          <Cloud />
          <AlertTitle>{cloudNotice(model)}</AlertTitle>
          <AlertDescription>
            <span>
              <span className="num">{model.id}</span> is served by the provider{' '}
              <span className="num">{model.provider?.name ?? 'unknown'}</span>: what you type here is sent there and
              billed by it.
            </span>
          </AlertDescription>
        </Alert>
      )}
      {model && !model.installed && (
        <Alert className="mb-4">
          <Info />
          <AlertTitle>
            <span className="num">{model.id}</span> is not installed
          </AlertTitle>
          <AlertDescription>{notInstalledHint(info.data, model)}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Text" description={`${text.trim().length} characters`}>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              aria-label="Text to speak"
              placeholder="Type something to say…"
              className="min-h-32 resize-y text-[15px] leading-7"
              lang={/[ऀ-ॿ]/.test(text) ? 'hi' : undefined}
              spellCheck={!ssml}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !busy) speak()
              }}
            />
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Languages className="mr-0.5 size-3.5 text-muted-foreground" aria-hidden />
              <span className="mr-1 text-xs text-muted-foreground">Try</span>
              {texts.map((s) => (
                <Button
                  key={s.id}
                  size="xs"
                  variant={text === s.text ? 'secondary' : 'ghost'}
                  onClick={() => setText(s.text)}
                  title={s.text}
                  className="border border-border"
                >
                  {languageName(s.language)}
                  {texts.filter((x) => x.language === s.language).length > 1 && (
                    <span className="text-muted-foreground">
                      {texts.filter((x) => x.language === s.language).indexOf(s) + 1}
                    </span>
                  )}
                </Button>
              ))}
            </div>
            {steerable && (
              <div className="mt-3 grid gap-1.5">
                <label htmlFor="tts-instructions" className="text-xs text-muted-foreground">
                  Instructions: how to say it (style, emotion, the voice), in words
                </label>
                <Textarea
                  id="tts-instructions"
                  value={instructions}
                  onChange={(e) => setInstructions(e.target.value)}
                  placeholder="Calm and warm, a little amused, speaking slowly."
                  className="min-h-14 resize-y text-[13px]"
                  maxLength={4096}
                  disabled={busy}
                  data-testid="tts-instructions"
                />
              </div>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {!isDemo && (
                <label
                  className="flex h-8 items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
                  title="Highlight each word as it is spoken: the audio is aligned to the text on the server"
                >
                  Word captions
                  <Switch
                    checked={captions}
                    onCheckedChange={setCaptions}
                    size="sm"
                    aria-label="Highlight each word as it is spoken"
                    data-testid="tts-captions"
                  />
                </label>
              )}
              <label className="flex h-8 items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30">
                SSML
                <Switch
                  checked={ssml}
                  onCheckedChange={setSsml}
                  size="sm"
                  disabled={busy}
                  aria-label="The text is SSML"
                  data-testid="tts-ssml"
                />
              </label>
              {multilingual && !lang && (
                <label
                  className="flex h-8 items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
                  title="Without a language, guess it from the text and, unless you named a voice, pick a voice that speaks it"
                >
                  Detect language
                  <Switch
                    checked={detect}
                    onCheckedChange={setDetect}
                    size="sm"
                    disabled={busy}
                    aria-label="Detect the language from the text"
                    data-testid="tts-detect-language"
                  />
                </label>
              )}
              {wordLists.offered && (
                <WordListSelect names={listNames} value={chosenLists} onChange={setLists} disabled={busy} />
              )}
              {romanisedCap && (
                <Select
                  value={roman || 'default'}
                  onValueChange={(v) => setRomanised(v === 'default' ? '' : v)}
                  disabled={busy}
                >
                  <SelectTrigger
                    className="h-8 w-56"
                    aria-label="Romanised text"
                    data-testid="tts-romanised"
                    title="Text typed in Latin letters (mujhe kal jana hai) is written in the voice's script before it is read"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" className="max-w-80">
                    <SelectItem value="default">
                      Romanised text: {romanisedPin ? 'pinned' : 'default'} (
                      {ROMANISED_LABEL[romanisedDefault as RomanisedMode] ?? romanisedDefault})
                    </SelectItem>
                    {ROMANISED_MODES.map((m) => (
                      <SelectItem key={m} value={m}>
                        <span className="flex flex-col">
                          <span>Romanised text: {ROMANISED_LABEL[m]}</span>
                          <span className="text-xs text-muted-foreground">{ROMANISED_HINT[m]}</span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {romanisedCap && romanisedNote && (roman === 'auto' || roman === 'on') && (
                <span className="text-xs text-warning-text" data-testid="tts-romanised-note">
                  {romanisedNote}
                </span>
              )}
              {multilingual && !lang && detect && !forcedHttp && transport === 'websocket' && (
                <span className="text-xs text-muted-foreground" data-testid="tts-detect-note">
                  The WebSocket does not detect the language; chunked HTTP does.
                </span>
              )}
              {forcedHttp && (
                <span className="text-xs text-muted-foreground">
                  {ssml ? 'SSML' : 'A cloned voice'} is sent over chunked HTTP.
                </span>
              )}
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
                  disabled={!model?.installed || !text.trim() || blocked || !cloneReady}
                  data-testid="speak"
                >
                  <Play className="fill-current" /> Speak
                </Button>
              )}
              <Button
                variant="outline"
                className="h-9"
                onClick={roundTrip}
                disabled={!model?.installed || !text.trim() || busy || blocked || trip?.status === 'running'}
                data-testid="round-trip"
              >
                <Repeat2 /> Speak, then transcribe
              </Button>
              <Button variant="ghost" className="h-9" onClick={save} disabled={!hasAudio}>
                <Download /> WAV
              </Button>
              <span className="ml-auto hidden text-xs text-muted-foreground sm:inline">
                <kbd className="rounded border border-border bg-muted px-1 num text-[10px]">Ctrl</kbd>{' '}
                <kbd className="rounded border border-border bg-muted px-1 num text-[10px]">Enter</kbd> to speak
              </span>
            </div>
          </Panel>

          {canClone(model) && (
            <VoiceClonePanel
              enabled={cloning}
              onEnabledChange={(on) => {
                setCloning(on)
                if (on) setSavedId(null)
              }}
              reference={reference}
              onReferenceChange={setReference}
              onSave={
                library.offered
                  ? (r) =>
                      setToSave({
                        blob: dataUrlToBlob(r.dataUrl),
                        name: r.label.includes('.') ? r.label : 'recording.wav',
                        seconds: r.seconds,
                        transcript: r.text,
                      })
                  : undefined
              }
              disabled={busy}
            />
          )}

          {blends && model && (
            <VoiceBlendPanel
              model={model.id}
              voices={voices}
              enabled={blending}
              onEnabledChange={(on) => {
                setBlending(on)
                if (on) {
                  setBlendId(null)
                  setSavedId(null)
                }
              }}
              parts={blendParts}
              onPartsChange={setParts}
              canSave={library.offered}
              onSaved={(b) => {
                setBlending(false)
                setBlendId(b.id)
              }}
              disabled={busy}
            />
          )}

          {library.offered && (
            <VoiceLibrary
              model={model?.id ?? null}
              canUse={canClone(model) && model?.installed === true}
              selected={saved?.id ?? null}
              onSelect={(id) => {
                setSavedId(id)
                if (id) setCloning(false)
              }}
              pending={toSave}
              onPendingClear={() => setToSave(null)}
              canBlend={blends && model?.installed === true}
              selectedBlend={savedBlend?.id ?? null}
              onSelectBlend={(id) => {
                setBlendId(id)
                if (id) {
                  setBlending(false)
                  setSavedId(null)
                }
              }}
              disabled={busy}
            />
          )}

          <Panel
            title="Audio"
            description={
              state.status === 'idle'
                ? 'Nothing synthesised yet'
                : `${state.status === 'connecting' ? 'Connecting' : state.status === 'synthesizing' ? 'Receiving and playing' : state.status === 'playing' ? 'Playing' : state.status === 'error' ? 'Failed' : 'Done'} · ${fmtSeconds(state.audioSeconds)}${state.sampleRate ? ` · ${(state.sampleRate / 1000).toFixed(state.sampleRate % 1000 ? 2 : 0)} kHz` : ''}`
            }
            actions={
              state.status !== 'idle' ? (
                <span
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 text-[11px]',
                    underrun ? 'border-warning/50 text-foreground' : 'border-border text-muted-foreground',
                  )}
                  data-testid="underruns"
                >
                  {underrun ? (
                    <TriangleAlert className="size-3 text-warning-text" aria-hidden />
                  ) : (
                    <CircleCheck className="size-3 text-good-text" aria-hidden />
                  )}
                  {underrun ? (
                    <>
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
              <ChunkTimeline chunks={state.chunks} className="mt-4" />
            ) : (
              <p className="mt-3 text-xs text-muted-foreground">
                Audio is scheduled back to back as it streams in. If playback ever catches up with synthesis, that gap
                is counted as an underrun.
              </p>
            )}
          </Panel>

          {captions && session && spoken && !isDemo && (
            <SpeechCaptions session={session} text={spoken} language={lang || undefined} />
          )}
          {trip && <RoundTripPanel trip={trip} />}
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Latency" description="Measured in this browser from sending the text">
            {state.status === 'idle' ? (
              <Empty title="No measurement yet" className="py-6">
                Press Speak. Time to first audio and the real-time factor appear as the stream arrives.
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
                    label="Server-reported first audio"
                    value={state.serverTtfaMs != null ? Math.round(state.serverTtfaMs) : '–'}
                    unit="ms"
                    hint={
                      state.ttfaMs != null && state.serverTtfaMs != null
                        ? `${fmtMs(Math.max(0, state.ttfaMs - state.serverTtfaMs))} network and client`
                        : transport === 'http'
                          ? 'not sent over HTTP'
                          : undefined
                    }
                  />
                  <Readout
                    label="Real-time factor"
                    value={fmtRtf(state.rtf)}
                    hint={state.rtf ? `${fmtSpeed(state.rtf)} real time` : 'when synthesis completes'}
                  />
                  <Readout
                    label="Audio"
                    value={state.audioSeconds.toFixed(2)}
                    unit="s"
                    hint={`${state.chunks.length} chunk${state.chunks.length === 1 ? '' : 's'}`}
                  />
                  <Readout
                    label="All audio received"
                    value={state.totalMs != null ? Math.round(state.totalMs) : '–'}
                    unit="ms"
                    hint={state.serverRtf != null ? `server RTF ${fmtRtf(state.serverRtf)}` : undefined}
                  />
                </div>
              </>
            )}
          </Panel>

          {state.status !== 'idle' && (
            <Panel
              title="What the server decided"
              description={state.transport === 'http' ? 'From the response headers' : 'Over the WebSocket'}
              data-testid="tts-decided"
            >
              {state.transport !== 'http' ? (
                <p className="text-[13px] leading-5 text-muted-foreground">
                  The WebSocket reports no headers: switch the transport to chunked HTTP to see the voice, blend and
                  language the server used (and to have the language detected from the text).
                </p>
              ) : !reported ? (
                <p className="text-[13px] leading-5 text-muted-foreground">Waiting for the response.</p>
              ) : (
                <div className="grid gap-3">
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                    {reported.model && (
                      <>
                        <dt className="text-muted-foreground">Model</dt>
                        <dd className="text-right num break-all">{reported.model}</dd>
                      </>
                    )}
                    {reported.voice && (
                      <>
                        <dt className="text-muted-foreground">Saved voice</dt>
                        <dd className="text-right num break-all">{reported.voice}</dd>
                      </>
                    )}
                    {reported.blend && (
                      <>
                        <dt className="text-muted-foreground">Blend</dt>
                        <dd className="text-right num break-all" data-testid="tts-blend-header">
                          {reported.blend}
                        </dd>
                      </>
                    )}
                    <dt className="text-muted-foreground">Language</dt>
                    <dd className="text-right" data-testid="tts-language-header">
                      {decision ? languageSummary(decision) : 'not reported'}
                    </dd>
                  </dl>
                  {reported.fallback && (
                    <p className="text-xs text-warning-text">Another model answered: {reported.fallback}</p>
                  )}
                  {reported.instructions === 'ignored' && (
                    <p className="text-xs text-warning-text" data-testid="tts-instructions-ignored">
                      The model that answered does not follow instructions; they were ignored.
                    </p>
                  )}
                  {decisionWarning && (
                    <Alert data-testid="tts-language-mismatch">
                      <TriangleAlert />
                      <AlertTitle>The voice may not speak this text&apos;s language</AlertTitle>
                      <AlertDescription>{decisionWarning}</AlertDescription>
                    </Alert>
                  )}
                  {decision && (
                    <p className="flex items-center gap-1.5 num text-[11px] text-muted-foreground">
                      <Tag className="size-3" aria-hidden /> x-bolo-language: {reported.language}
                    </p>
                  )}
                </div>
              )}
            </Panel>
          )}

          <Panel title="Reading the numbers" bodyClassName="grid gap-2.5 text-[13px] leading-5 text-muted-foreground">
            <p>
              <span className="text-foreground">Time to first audio</span> is what a caller waits before hearing
              anything. It matters far more than total synthesis time.
            </p>
            <p>
              <span className="text-foreground">Real-time factor</span> below 1 means audio is produced faster than it
              plays, so playback never has to wait once it starts.
            </p>
            <p>
              <span className="text-foreground">Headroom</span> in the chunk timeline is how long each chunk sat ready
              before it was needed.
            </p>
          </Panel>
        </div>
      </div>
    </Page>
  )
}
