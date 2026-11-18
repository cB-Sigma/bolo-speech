import { Ear, Hourglass, TimerOff } from 'lucide-react'
import { CatalogSelect } from '@/components/catalog-select'
import { Panel } from '@/components/panel'
import { OptionRow } from '@/components/stt/browser-processing'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useModels } from '@/hooks/use-api'
import type { ServerInfo } from '@/lib/api/types'
import { autoPull, readsText } from '@/lib/capabilities'
import type { ConversationState, Listening } from '@/lib/realtime/session'

/** Choices for `max_turn_seconds` besides the server's own (0: no limit). */
const TURN_LIMITS = [15, 30, 60, 0] as const

/**
 * How a Realtime session on the server hears the end of a turn (`audio.input.turn_detection`): the voice-activity
 * detector (`vad_model`) finds the pauses; with `semantic_vad` an end-of-turn model (`turn_model`) is asked at each
 * pause whether the speaker has finished. Every catalog model is offered; one that is not installed is downloaded on
 * first use when the server pulls models by itself.
 */
export function ServerListeningPanel({
  info,
  value,
  onChange,
  disabled,
  reported,
  className,
}: {
  info: ServerInfo | undefined
  value: Listening
  onChange: (l: Listening) => void
  disabled?: boolean
  /** What the server says the session uses, once it is live. */
  reported?: ConversationState['serverListening']
  className?: string
}) {
  const vads = useModels('vad').models
  const turns = useModels('turn').models
  const pull = autoPull(info)
  const defaultVad = info?.defaults?.vad
  const semantic = info?.capabilities?.realtime?.semantic_vad
  const limits = info?.capabilities?.realtime?.turn_limits
  const chosenTurn = turns.find((m) => m.id === value.turnModel)
  const textTurn = chosenTurn ? readsText(chosenTurn) : !value.turnModel && semantic?.input === 'text'
  return (
    <Panel
      title="Listening"
      description="Voice activity and end of turn for this session (turn_detection)"
      className={className}
      data-testid="server-listening"
      bodyClassName="divide-y divide-border"
    >
      <OptionRow
        testId="server-vad-option"
        icon={<Ear />}
        title="Voice activity"
        hint={
          <>
            What counts as speech (<span className="font-mono">vad_model</span>). On the server’s noisy test stream:
            Silero v5 and v6 0.93 frame accuracy, MarbleNet 0.92, Silero v4 (the default) 0.89, WebRTC 0.85, TEN VAD
            0.78, the energy detector 0.74.
          </>
        }
        control={
          <CatalogSelect
            models={vads}
            value={value.vadModel ?? ''}
            onChange={(id) => onChange({ ...value, vadModel: id })}
            label="Voice-activity detector"
            defaultLabel={defaultVad ? `Server default (${defaultVad})` : 'Server default'}
            disabled={disabled}
            autoPull={pull}
            testId="server-vad"
          />
        }
      />
      <OptionRow
        testId="server-turn-option"
        icon={<Hourglass />}
        title="End of turn"
        hint={
          <>
            <span className="font-mono">server_vad</span>: a pause ends the turn.{' '}
            <span className="font-mono">semantic_vad</span>: at each pause an end-of-turn model says whether you have
            finished; an unfinished turn waits up to 3 s. Models that read the transcript decode the utterance at each
            pause with the session’s recogniser (not with cloud recognisers).
          </>
        }
        control={
          <div className="grid justify-items-end gap-1.5">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={value.type}
              onValueChange={(v) => v && onChange({ ...value, type: v as Listening['type'] })}
              aria-label="Turn detection"
              disabled={disabled}
              data-testid="turn-detection-type"
            >
              <ToggleGroupItem value="server_vad" className="px-2.5 text-xs">
                Pause
              </ToggleGroupItem>
              <ToggleGroupItem value="semantic_vad" className="px-2.5 text-xs" data-testid="turn-detection-semantic">
                Pause + model
              </ToggleGroupItem>
            </ToggleGroup>
            {value.type === 'semantic_vad' && (
              <CatalogSelect
                models={turns}
                value={value.turnModel ?? ''}
                onChange={(id) => onChange({ ...value, turnModel: id })}
                label="End-of-turn model"
                defaultLabel={semantic?.model ? `Server default (${semantic.model})` : 'Server default'}
                disabled={disabled}
                autoPull={pull}
                extra={(m) =>
                  readsText(m) ? 'reads the transcript' : m.id === 'silence-only' ? 'no model' : 'hears the audio'
                }
                testId="server-turn"
              />
            )}
          </div>
        }
      >
        {value.type === 'semantic_vad' && textTurn && (
          <p className="pl-8.5 text-xs leading-4 text-muted-foreground" data-testid="text-turn-note">
            This model reads text: each pause costs one extra decode of what you said so far.
          </p>
        )}
      </OptionRow>
      <OptionRow
        testId="server-turn-limit-option"
        icon={<TimerOff />}
        title="Longest turn"
        hint={
          <>
            A fan or other steady noise can keep the detector hearing “speech”, so your turn never ends and the agent
            never answers. The turn is committed after this long anyway (
            <span className="font-mono">max_turn_seconds</span>), and, with the steady-noise check, as soon as the sound
            stops rising and falling the way a voice does (<span className="font-mono">noise_plateau_ms</span>,{' '}
            {limits?.noise_plateau_ms != null ? `${limits.noise_plateau_ms / 1000} s` : 'about 2 s'} on this server).
            Turn the check off if a quiet voice in a loud room is cut short.
          </>
        }
        control={
          <div className="grid justify-items-end gap-1.5">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={value.maxTurnSeconds === undefined ? 'default' : String(value.maxTurnSeconds)}
              onValueChange={(v) =>
                v && onChange({ ...value, maxTurnSeconds: v === 'default' ? undefined : Number(v) })
              }
              aria-label="Longest turn"
              disabled={disabled}
              data-testid="max-turn"
            >
              <ToggleGroupItem value="default" className="px-2.5 text-xs" title="The server's max_turn_seconds">
                {limits?.max_turn_seconds ? `Server (${limits.max_turn_seconds} s)` : 'Server'}
              </ToggleGroupItem>
              {TURN_LIMITS.map((n) => (
                <ToggleGroupItem key={n} value={String(n)} className="px-2.5 text-xs" data-testid={`max-turn-${n}`}>
                  {n === 0 ? 'No limit' : `${n} s`}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              <Switch
                checked={value.noisePlateau !== false}
                onCheckedChange={(on) => onChange({ ...value, noisePlateau: on ? undefined : false })}
                disabled={disabled}
                aria-label="Steady-noise check"
                data-testid="noise-plateau"
              />
              Steady-noise check
            </label>
          </div>
        }
      />
      {reported && (
        <p className="pt-2 text-xs text-muted-foreground" data-testid="server-listening-reported">
          Server: <span className="num">{reported.type}</span>
          {reported.vad_model ? (
            <>
              {' '}
              · detector <span className="num">{reported.vad_model}</span>
            </>
          ) : null}
          {reported.max_turn_seconds != null ? (
            <>
              {' '}
              · longest turn{' '}
              <span className="num">{reported.max_turn_seconds ? `${reported.max_turn_seconds} s` : 'no limit'}</span>
              {reported.noise_plateau_ms === 0 ? ', no steady-noise check' : ''}
            </>
          ) : null}
          {reported.type === 'semantic_vad' && reported.turn_model ? (
            <>
              {' '}
              · end of turn <span className="num">{reported.turn_model}</span>
            </>
          ) : null}
        </p>
      )}
    </Panel>
  )
}
