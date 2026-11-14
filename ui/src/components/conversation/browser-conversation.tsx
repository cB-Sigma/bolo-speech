import { Bot, Mic, PhoneOff, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { DownloadsPanel } from '@/components/downloads-panel'
import { Page } from '@/components/layout/app-shell'
import { useDevice, VerdictBadge } from '@/components/model-fit'
import { Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { openMicrophone, type AudioSource } from '@/lib/audio/capture'
import { GaplessPlayer } from '@/lib/audio/player'
import { Conversation, type Speaker } from '@/lib/browser-llm/conversation'
import { EndpointLlm, readEndpoint, saveEndpoint, sharedLlm, type Endpoint } from '@/lib/browser-llm/engine'
import { LLM_MODELS } from '@/lib/browser-llm/models'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { sharedTtsEngine } from '@/lib/browser-tts/engine'
import { prepareText } from '@/lib/browser-tts/text'
import { isPhoneLike, verdictFor } from '@/lib/device'
import { isDownloaded } from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { BrowserTranscriber } from '@/lib/stt/browser'
import { cn } from '@/lib/utils'

/** Speaks with the voice chosen on the Text to speech screen, into one gapless player. */
function voiceSpeaker(player: GaplessPlayer): Speaker {
  const tts = sharedTtsEngine()
  const fake = (() => {
    try {
      return Boolean(globalThis.localStorage?.getItem('bolo.mock.tts'))
    } catch {
      return false
    }
  })()
  return {
    async say(sentence, signal) {
      if (fake) {
        // Tests: a short tone per sentence instead of a voice.
        const sr = 16000
        const tone = Float32Array.from({ length: sr * 0.4 }, (_, i) => 0.1 * Math.sin((2 * Math.PI * 440 * i) / sr))
        if (!signal.aborted) player.enqueue(tone, sr)
        return
      }
      await tts.load()
      const cfg = tts.store.get().config
      if (!cfg || signal.aborted) return
      for (const chunk of prepareText(sentence, cfg, tts.store.get().voice.numbers).chunks) {
        const r = await tts.synthesize(chunk)
        if (signal.aborted) return
        player.enqueue(r.audio, r.sampleRate)
      }
    },
    stop() {
      tts.cancelPending()
      player.stop()
    },
    async drained() {
      while (player.bufferedAhead > 0.05) await new Promise((r) => setTimeout(r, 100))
    },
  }
}

const STATUS = {
  idle: 'Not started',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Speaking',
  error: 'Stopped by an error',
} as const

/**
 * The live demo's voice conversation: microphone → recogniser → language model → voice, all in the page,
 * or with the visitor's own OpenAI-compatible endpoint for the language model. Talking over the reply stops it.
 */
export function BrowserConversation() {
  const llm = sharedLlm()
  const ls = useStore(llm.store)
  const stt = useStore(sharedEngine().store)
  const tts = useStore(sharedTtsEngine().store)
  const d = useDevice()
  const [source, setSource] = useState<'page' | 'endpoint'>(() => (readEndpoint().url ? 'endpoint' : 'page'))
  const [endpoint, setEndpoint] = useState<Endpoint>(readEndpoint)
  const [conv, setConv] = useState<Conversation | null>(null)
  const state = useStore(conv?.store ?? idleStore)
  const parts = useRef<{
    mic: AudioSource
    transcriber: BrowserTranscriber
    player: GaplessPlayer
    off: () => void
  } | null>(null)
  const [starting, setStarting] = useState(false)
  const [live, setLive] = useState(false)
  const gpu = d.webgpu
  const onGpu = Boolean(gpu?.available && gpu.f16)

  const stop = async () => {
    const p = parts.current
    parts.current = null
    setLive(false)
    conv?.interrupt()
    if (!p) return
    p.off()
    await p.mic.stop().catch(() => undefined)
    p.transcriber.cancel()
    await p.player.close().catch(() => undefined)
    setConv((c) => {
      c?.store.patch({ status: 'idle' })
      return c
    })
  }
  useEffect(
    () => () => {
      void stop()
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const [typed, setTyped] = useState('')
  const ensure = () => {
    if (conv) return conv
    const player = new GaplessPlayer()
    const c = new Conversation(
      () => (source === 'endpoint' ? new EndpointLlm(readEndpoint()) : llm),
      voiceSpeaker(player),
    )
    setConv(c)
    return c
  }
  const send = () => {
    const text = typed.trim()
    if (!text) return
    setTyped('')
    void ensure().respond(text, null)
  }

  const start = async () => {
    setStarting(true)
    // The player is made in the tap, so audio may play later without another gesture.
    const player = new GaplessPlayer()
    const c =
      conv ??
      new Conversation(() => (source === 'endpoint' ? new EndpointLlm(readEndpoint()) : llm), voiceSpeaker(player))
    try {
      const transcriber = new BrowserTranscriber(sharedEngine(), { language: 'auto', endpointMs: 600 })
      const off = transcriber.store.subscribe(() => c.onTranscript(transcriber.store.get()))
      const mic = await openMicrophone((b) => transcriber.push(b))
      await transcriber.start()
      if (source === 'page') void llm.load().catch(() => undefined)
      parts.current = { mic, transcriber, player, off }
      setLive(true)
      c.store.patch({ status: 'listening', error: null })
      setConv(c)
    } catch (err) {
      c.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      setConv(c)
      await player.close().catch(() => undefined)
    } finally {
      setStarting(false)
    }
  }

  return (
    <Page title="Conversation">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="grid min-w-0 content-start gap-4">
          <Panel
            title="Talk to it"
            description="Microphone → speech to text → language model → voice, in this page. Talk over it to interrupt."
            data-testid="browser-conversation"
            actions={
              <div className="flex items-center gap-2">
                {conv && !live && (
                  <Button size="sm" variant="ghost" onClick={() => conv.reset()} aria-label="Clear the conversation">
                    <RotateCcw />
                  </Button>
                )}
                {live ? (
                  <Button size="sm" variant="destructive" onClick={() => void stop()} data-testid="conversation-stop">
                    <PhoneOff /> End
                  </Button>
                ) : (
                  <Button size="sm" onClick={() => void start()} disabled={starting} data-testid="conversation-start">
                    <Mic /> {starting ? 'Starting…' : 'Start talking'}
                  </Button>
                )}
              </div>
            }
          >
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs" data-testid="conversation-status">
              <Badge variant="outline" data-status={state.status}>
                {STATUS[state.status]}
              </Badge>
              {state.bargeIns > 0 && (
                <span className="text-muted-foreground" data-testid="barge-ins">
                  interrupted {state.bargeIns}×
                </span>
              )}
              {state.partial && <span className="truncate text-muted-foreground italic">{state.partial}</span>}
            </div>
            {state.error && (
              <p className="mb-3 text-xs text-destructive" data-testid="conversation-error">
                {state.error}
              </p>
            )}
            <form
              className="mb-3 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                send()
              }}
            >
              <Input
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                placeholder="Or type a message"
                aria-label="Type a message"
                data-testid="conversation-type"
              />
              <Button
                type="submit"
                size="sm"
                variant="outline"
                disabled={!typed.trim()}
                data-testid="conversation-send"
              >
                Send
              </Button>
            </form>
            {state.turns.length === 0 ? (
              <p className="text-[13px] text-muted-foreground">
                Press Start talking and say something. The reply is spoken sentence by sentence as it is written.
              </p>
            ) : (
              <ol className="grid gap-2" data-testid="conversation-turns">
                {state.turns.map((t, i) => (
                  <li
                    key={i}
                    className={cn(
                      'max-w-[90%] rounded-lg px-3 py-2 text-[13px] leading-5',
                      t.role === 'user' ? 'justify-self-end bg-muted' : 'border border-border',
                    )}
                    data-role={t.role}
                  >
                    <span data-testid="turn-text">{t.text || <span className="text-muted-foreground">…</span>}</span>
                    {t.interrupted && <span className="text-muted-foreground"> (interrupted)</span>}
                    {t.latency && (
                      <div className="mt-1 font-mono text-[11px] text-muted-foreground" data-testid="turn-latency">
                        speech→text {t.latency.sttMs ?? '–'} ms · text→first word {t.latency.firstTokenMs ?? '–'} ms ·
                        text→first sound {t.latency.firstAudioMs ?? '–'} ms
                      </div>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </Panel>
          <Panel title="Speech to speech in one model" data-testid="s2s-note">
            <p className="text-xs text-muted-foreground">
              LFM3-Audio 2B (LFM Open License v1.0, about 2.7 GB as ONNX) answers speech with speech in one model on a
              bolo server. No browser runtime runs it yet, so it is not offered here; the pipeline above is the in-page
              way to talk to a model.
            </p>
          </Panel>
        </div>
        <div className="grid min-w-0 content-start gap-4">
          <Panel title="Language model" data-testid="llm-settings">
            <div className="grid gap-3 text-xs">
              <Select value={source} onValueChange={(v) => setSource(v as 'page' | 'endpoint')} disabled={live}>
                <SelectTrigger
                  className="h-8 w-full"
                  aria-label="Where the language model runs"
                  data-testid="llm-source"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="page">In this page</SelectItem>
                  <SelectItem value="endpoint">My own OpenAI-compatible endpoint</SelectItem>
                </SelectContent>
              </Select>
              {source === 'page' ? (
                <>
                  <Select value={ls.model.id} onValueChange={(id) => llm.setModel(id)} disabled={live}>
                    <SelectTrigger className="h-8 w-full" aria-label="Language model" data-testid="llm-model">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent position="popper" className="w-[min(24rem,calc(100vw-2rem))]">
                      {LLM_MODELS.map((m) => (
                        <SelectItem key={m.id} value={m.id} textValue={m.label} data-testid={`llm-${m.id}`}>
                          <span className="grid">
                            <span>{m.label}</span>
                            <span className="text-[11px] text-muted-foreground">
                              {fmtBytes(onGpu ? m.bytes.q4f16 : m.bytes.q4)} · {m.license}
                              {m.gpuOnly ? ' · needs WebGPU, laptop' : ''}
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="flex flex-wrap items-center gap-2">
                    <VerdictBadge
                      verdict={verdictFor(
                        {
                          bytes: onGpu ? ls.model.bytes.q4f16 : ls.model.bytes.q4,
                          peakBytes: (onGpu ? ls.model.bytes.q4f16 : ls.model.bytes.q4) * 1.3 + 150e6,
                          requiresWebGpu: ls.model.gpuOnly,
                          downloaded: isDownloaded(`llm-${ls.model.id}`),
                        },
                        d,
                      )}
                    />
                    <span className="text-muted-foreground" data-testid="llm-about">
                      {ls.model.params} parameters, 4-bit{' '}
                      {onGpu ? 'on WebGPU' : 'on the CPU (no WebGPU with 16-bit shaders here: slower)'}.{' '}
                      {ls.model.blurb} {ls.model.licenseNote ?? ''}
                    </span>
                  </div>
                  <p data-testid="llm-status">
                    {ls.status === 'loading'
                      ? `Downloading · ${Math.round(100 * ls.progress)}%`
                      : ls.status === 'ready'
                        ? `Ready on ${ls.device === 'webgpu' ? 'WebGPU' : 'the CPU'}`
                        : ls.status === 'error'
                          ? `Could not load: ${ls.error}`
                          : 'Downloaded when you start'}
                  </p>
                </>
              ) : (
                <form
                  className="grid gap-2"
                  onSubmit={(e) => {
                    e.preventDefault()
                    saveEndpoint(endpoint)
                  }}
                  data-testid="endpoint-form"
                >
                  <label className="grid gap-1">
                    <span className="text-muted-foreground">Base URL</span>
                    <Input
                      value={endpoint.url}
                      placeholder="https://api.example.com/v1"
                      onChange={(e) => setEndpoint({ ...endpoint, url: e.target.value })}
                      onBlur={() => saveEndpoint(endpoint)}
                    />
                  </label>
                  <label className="grid gap-1">
                    <span className="text-muted-foreground">Model</span>
                    <Input
                      value={endpoint.model}
                      onChange={(e) => setEndpoint({ ...endpoint, model: e.target.value })}
                      onBlur={() => saveEndpoint(endpoint)}
                    />
                  </label>
                  <label className="grid gap-1">
                    <span className="text-muted-foreground">API key</span>
                    <Input
                      type="password"
                      autoComplete="off"
                      value={endpoint.key}
                      onChange={(e) => setEndpoint({ ...endpoint, key: e.target.value })}
                      onBlur={() => saveEndpoint(endpoint)}
                    />
                  </label>
                  <p className="text-muted-foreground">
                    Kept only in this browser&apos;s storage and sent only to that URL, straight from this page. The
                    endpoint must allow requests from this site (CORS).
                  </p>
                </form>
              )}
            </div>
          </Panel>
          <Panel title="Ears and voice" data-testid="conversation-models">
            <ul className="grid gap-1 text-xs">
              <li>
                <Mic className="mr-1 inline size-3.5" aria-hidden /> Speech to text: {stt.model.label} (change it on
                Speech to text)
              </li>
              <li>
                <Bot className="mr-1 inline size-3.5" aria-hidden /> Voice: {tts.voice.label} (change it on Text to
                speech)
              </li>
            </ul>
            {isPhoneLike() && (
              <p className="mt-2 text-xs text-warning-text" data-testid="phone-note">
                On a phone only one model fits in the page at a time: expect pauses while the recogniser, the language
                model and the voice take turns. Your own endpoint for the language model helps.
              </p>
            )}
          </Panel>
          <DownloadsPanel />
        </div>
      </div>
    </Page>
  )
}

const idleStore = new Conversation(() => ({ reply: async () => '' }), {
  say: async () => undefined,
  stop: () => undefined,
  drained: async () => undefined,
}).store
