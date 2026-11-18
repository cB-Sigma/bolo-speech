import { Info, MessagesSquare, Mic, MicOff, PhoneOff, Square } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { BrowserConversation } from '@/components/conversation/browser-conversation'
import { PipelineModelPanel } from '@/components/conversation/pipeline-model'
import { PipelinePicker } from '@/components/conversation/pipeline-picker'
import { ServerListeningPanel } from '@/components/conversation/server-listening'
import { defaultEcho, EchoControl, realtimeEcho, type Echo } from '@/components/conversation/echo-control'
import { AgentToolsPanel, mcpTools, noMcp, type McpChoice } from '@/components/conversation/agent-tools'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { MicHelpPanel } from '@/components/stt/mic-help'
import { DenoiseControl } from '@/components/stt/transcribe-options'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useComponents, useInfo } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { useSupport } from '@/hooks/use-support'
import { isDemo } from '@/lib/api/config'
import { defaultPipeline, hasPipelines, pipelineOptions, type PipelineId } from '@/lib/capabilities'
import { fmtMs } from '@/lib/format'
import {
  ConversationSession,
  emptyConversation,
  type ConversationState,
  type Listening,
  type RealtimeTransport,
} from '@/lib/realtime/session'
import { Store } from '@/lib/store'
import { defaultDenoise, realtimeDenoise, type Denoise } from '@/lib/stt-options'
import { cn } from '@/lib/utils'

const idle = new Store<ConversationState>(emptyConversation())

const ERROR_TITLE: Record<string, string> = {
  engine_unavailable: 'The model is not reachable',
  webrtc_unavailable: 'WebRTC is not installed on the server',
  webrtc_failed: 'The WebRTC connection failed',
  unauthorized: 'The server refused the API key',
}

const STATUS: Record<ConversationState['status'], string> = {
  idle: 'Not started',
  starting: 'Connecting',
  live: 'Live',
  ended: 'Ended',
  error: 'Stopped by an error',
}

export default function ConversationPage() {
  return isDemo ? <BrowserConversation /> : <ServerConversation />
}

function ServerConversation() {
  const info = useInfo()
  const components = useComponents().data?.components
  const support = useSupport()
  const options = useMemo(() => pipelineOptions(info.data, components ?? []), [info.data, components])
  const [chosen, setChosen] = useState<PipelineId | null>(null)
  const pipeline = chosen ?? defaultPipeline(info.data, options)
  const [transport, setTransport] = useState<RealtimeTransport>('websocket')
  const [denoise, setDenoise] = useState<Denoise>(defaultDenoise)
  const [listening, setListening] = useState<Listening>({ type: 'server_vad' })
  const [mcp, setMcp] = useState<McpChoice>(noMcp)
  const [echo, setEcho] = useState<Echo>(defaultEcho)
  const [session, setSession] = useState<ConversationSession | null>(null)
  const state = useStore(session?.store ?? idle)
  const active = state.status === 'starting' || state.status === 'live'
  const selected = options.find((o) => o.id === pipeline)
  // A full-duplex model hears and speaks at once: talking over it does not stop it, the stop button does.
  const fullDuplex =
    pipeline === 'speech-to-speech' && Boolean(info.data?.capabilities?.realtime?.pipelines?.[pipeline]?.full_duplex)
  const webrtc = info.data?.capabilities?.realtime?.webrtc
  const webrtcAvailable = webrtc?.available !== false && typeof RTCPeerConnection !== 'undefined'
  const webrtcReason =
    typeof RTCPeerConnection === 'undefined'
      ? 'This browser has no WebRTC'
      : webrtc?.available === false
        ? `The server has no WebRTC transport: ${webrtc.install ?? "pip install 'bolo-speech[webrtc]'"}`
        : undefined
  const micBlocked =
    !support.getUserMedia || !support.secureContext || (transport === 'websocket' && !support.audioWorklet)

  useEffect(() => () => void session?.stop(), [session])

  const start = () => {
    // Everything that needs the tap happens inside start(): the microphone prompt first.
    const s = new ConversationSession(pipeline, transport, realtimeDenoise(denoise), {
      listening,
      fullDuplex,
      tools: pipeline === 'speech-to-speech' ? [] : mcpTools(mcp),
      echo: realtimeEcho(echo),
    })
    void s.start()
    setSession(s)
  }

  if (isDemo) {
    return (
      <Page title="Conversation">
        <NeedsServer title="Voice conversations need a bolo server">
          A conversation sends your voice to a speech server, which hears you, asks a language model (or a
          speech-to-speech model) for an answer and speaks it back. This demo runs in your browser only.
        </NeedsServer>
      </Page>
    )
  }

  const known = hasPipelines(info.data)
  const noneAvailable = !options.some((o) => o.available)

  return (
    <Page title="Conversation">
      {info.isLoading ? (
        <Skeleton className="mb-4 h-32 rounded-xl" />
      ) : (
        <Panel
          className="mb-4"
          title="Pipeline"
          description={
            known
              ? 'What answers you. Sent as session.pipeline before any audio, then fixed for the conversation.'
              : 'This server does not report pipelines; it uses its configured one.'
          }
        >
          <PipelinePicker options={options} value={pipeline} onChange={setChosen} locked={active} />
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
            <span id="transport-label">Transport</span>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={transport}
              onValueChange={(v) => v && setTransport(v as RealtimeTransport)}
              aria-labelledby="transport-label"
              disabled={active}
              data-testid="transport"
            >
              <ToggleGroupItem value="websocket" className="px-3 text-xs" data-testid="transport-websocket">
                WebSocket
              </ToggleGroupItem>
              <ToggleGroupItem
                value="webrtc"
                className="px-3 text-xs"
                disabled={!webrtcAvailable}
                title={webrtcReason}
                data-testid="transport-webrtc"
              >
                WebRTC
              </ToggleGroupItem>
            </ToggleGroup>
            <span>
              {transport === 'webrtc'
                ? 'Microphone and voice as WebRTC audio tracks, events on a data channel (POST /v1/realtime/calls).'
                : 'Audio as PCM in Realtime events over WS /v1/realtime.'}
            </span>
          </div>
          <DenoiseControl value={denoise} onChange={setDenoise} disabled={active} className="mt-3" />
          <EchoControl
            info={info.data}
            value={echo}
            onChange={setEcho}
            disabled={active}
            reported={state.serverEcho}
            live={state.status !== 'idle'}
            className="mt-3"
          />
          {state.status !== 'idle' && (
            <p className="mt-1.5 text-xs text-muted-foreground" data-testid="server-denoise">
              Server:{' '}
              {state.serverDenoise?.enabled
                ? `denoising with ${state.serverDenoise.model ?? 'the default denoiser'} at strength ${state.serverDenoise.strength}`
                : state.serverDenoise
                  ? 'no denoising'
                  : 'denoising as configured on the server (not reported)'}
            </p>
          )}
        </Panel>
      )}

      {noneAvailable && !info.isLoading && (
        <Alert className="mb-4">
          <Info />
          <AlertTitle>No pipeline is set up on this server</AlertTitle>
          <AlertDescription>
            Connect a language model (setting llm_base_url, or bolo serve --llm-base-url …) for the cascade, an
            audio-understanding model for audio-llm, or a speech-to-speech model (s2s_model with llama-server, or
            s2s_server_url). See docs/voice-agents.md, &quot;Pipelines&quot;.
          </AlertDescription>
        </Alert>
      )}

      {state.mic && (
        <MicHelpPanel className="mb-4" failure={state.mic} onRetry={start} onDismiss={() => setSession(null)} />
      )}
      {state.error && !state.mic && (
        <Alert className="mb-4" variant="destructive" data-testid="conversation-error">
          <Info />
          <AlertTitle>{ERROR_TITLE[state.errorCode ?? ''] ?? 'Error'}</AlertTitle>
          <AlertDescription>{state.error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Panel
          title="Conversation"
          description={`${STATUS[state.status]}${state.serverPipeline ? ` · pipeline ${state.serverPipeline}` : ''}`}
          actions={
            active ? (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => session?.stopAnswer()}
                  disabled={state.status !== 'live'}
                  title="Stops what the agent is saying (response.cancel); the conversation goes on"
                  data-testid="stop-answer"
                >
                  <Square className="fill-current" /> Stop the answer
                </Button>
                <Button variant="destructive" size="sm" onClick={() => void session?.stop()} data-testid="end-call">
                  <PhoneOff /> End
                </Button>
              </div>
            ) : (
              <Button
                size="sm"
                onClick={start}
                disabled={micBlocked || !selected?.available}
                title={
                  micBlocked
                    ? 'This browser cannot capture the microphone here'
                    : !selected?.available
                      ? (selected?.reason ?? undefined)
                      : undefined
                }
                data-testid="start-call"
              >
                <Mic /> Start talking
              </Button>
            )
          }
        >
          {fullDuplex && (
            <p className="mb-3 text-xs text-muted-foreground" data-testid="full-duplex-hint">
              Full duplex: talking over the model does not stop it. Stop the answer does.
            </p>
          )}
          <div className="mb-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground" aria-live="polite">
            <Indicator on={state.userSpeaking} label="You are speaking" idleLabel="You: quiet" />
            <Indicator on={state.agentSpeaking} label="Agent is speaking" idleLabel="Agent: quiet" />
            {state.transport === 'webrtc' && state.link && (
              <span
                className="num"
                data-testid="webrtc-state"
                data-connection={state.link.connection}
                data-heard={state.remoteAudio?.heard ? 'true' : 'false'}
                data-packets={state.remoteAudio?.packets ?? 0}
              >
                WebRTC {state.link.connection} · ICE {state.link.ice} · events {state.link.channel}
                {state.remoteAudio &&
                  ` · agent audio ${state.remoteAudio.seconds.toFixed(1)} s${state.remoteAudio.heard ? ', sound received' : ''}`}
              </span>
            )}
            {micBlocked && (
              <span className="inline-flex items-center gap-1 text-warning-text">
                <MicOff className="size-3.5" aria-hidden /> Microphone capture is unavailable here (HTTPS or localhost
                needed)
              </span>
            )}
          </div>
          {state.turns.length === 0 ? (
            <Empty icon={MessagesSquare} title={active ? 'Listening' : 'Nothing said yet'} className="py-8">
              {active
                ? 'Say something. Your words and the answer appear here as they are recognised and spoken.'
                : 'Pick a pipeline and press Start talking. The browser asks for the microphone when you press it.'}
            </Empty>
          ) : (
            <ol className="grid gap-2" data-testid="turns">
              {state.turns.map((t) => (
                <li
                  key={`${t.role}-${t.id}`}
                  className={cn(
                    'max-w-[85%] rounded-lg border px-3 py-2 text-[14px] leading-6',
                    t.role === 'user'
                      ? 'justify-self-end border-chart-1/30 bg-chart-1/10'
                      : 'border-border bg-muted/40',
                  )}
                >
                  <div className="mb-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
                    <span className="font-medium">{t.role === 'user' ? 'You' : 'Agent'}</span>
                    {t.firstAudioMs != null && <span className="num">first audio {fmtMs(t.firstAudioMs)}</span>}
                    {t.status && <span className="text-warning-text">{t.status}</span>}
                  </div>
                  <span className={cn(!t.final && 'text-muted-foreground')}>{t.text || '…'}</span>
                </li>
              ))}
            </ol>
          )}
        </Panel>

        <div className="grid min-w-0 content-start gap-4">
          <PipelineModelPanel key={pipeline} pipeline={pipeline} info={info.data} />
          <ServerListeningPanel
            info={info.data}
            value={listening}
            onChange={setListening}
            disabled={active}
            reported={state.serverListening}
          />
          {pipeline !== 'speech-to-speech' && (
            <AgentToolsPanel
              info={info.data}
              value={mcp}
              onChange={setMcp}
              disabled={active}
              state={state}
              onApprove={(id, approve) => session?.answerApproval(id, approve)}
            />
          )}
          <Panel title="About the pipelines" bodyClassName="grid gap-2.5 text-[13px] leading-5 text-muted-foreground">
            <p>
              <span className="text-foreground">Cascade</span> is speech to text, your language model, then text to
              speech. Every step is visible on the Latency screen.
            </p>
            <p>
              <span className="text-foreground">Audio LLM</span> lets a model that understands audio hear your turn
              itself, so it can pick up tone; bolo still speaks the answer.
            </p>
            <p>
              <span className="text-foreground">Speech to speech</span> hands the whole turn to a model that answers in
              its own voice. bolo keeps turn detection, interruptions and the transcripts.
            </p>
            <p>
              Interrupt the agent by talking: its audio stops at once (not a full-duplex model&apos;s: see its panel).
              The server default is{' '}
              <span className="num text-foreground">{info.data?.capabilities?.realtime?.pipeline ?? 'cascade'}</span>.
            </p>
          </Panel>
        </div>
      </div>
    </Page>
  )
}

function Indicator({ on, label, idleLabel }: { on: boolean; label: string; idleLabel: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden className={cn('size-2 rounded-full', on ? 'live-dot bg-chart-1' : 'bg-chart-mute')} />
      {on ? label : idleLabel}
    </span>
  )
}
