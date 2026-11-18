import { AudioLines } from 'lucide-react'
import { useId } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { Label } from '@/components/ui/label'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useModels } from '@/hooks/use-api'
import type { ServerInfo } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'
import type { ConversationState, RealtimeEcho } from '@/lib/realtime/session'
import { cn } from '@/lib/utils'

/** The console's choice: the server's setting, off, or on with a canceller ('' for the server's default one). */
export interface Echo {
  mode: 'default' | 'off' | 'on'
  model: string
}

export const defaultEcho: Echo = { mode: 'default', model: '' }

/** `audio.input.echo_cancellation` for a choice; undefined leaves the server's setting. */
export function realtimeEcho(e: Echo): RealtimeEcho | undefined {
  if (e.mode === 'default') return undefined
  if (e.mode === 'off') return false
  return e.model ? { model: e.model } : {}
}

/**
 * Server-side echo cancellation (`audio.input.echo_cancellation`): the agent's voice removed from what the session
 * hears, for clients without their own (phones, devices). The browser cancels echo on the microphone already.
 */
export function EchoControl({
  info,
  value,
  onChange,
  disabled,
  reported,
  live,
  className,
}: {
  info: ServerInfo | undefined
  value: Echo
  onChange: (e: Echo) => void
  disabled?: boolean
  reported?: ConversationState['serverEcho']
  /** The conversation has started: say what the server reports. */
  live?: boolean
  className?: string
}) {
  const id = useId()
  const models = useModels('echo-cancellation').models
  const caps = info?.capabilities?.realtime?.echo_cancellation
  if (!caps) return null
  return (
    <div className={cn('grid gap-1.5', className)} data-testid="echo-control">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs" id={`${id}-label`}>
          <AudioLines className="size-3.5 text-muted-foreground" aria-hidden /> Echo cancellation
        </Label>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={value.mode}
          onValueChange={(v) => v && onChange({ ...value, mode: v as Echo['mode'] })}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
        >
          <ToggleGroupItem value="default" className="px-2.5 text-xs">
            Server default
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="px-2.5 text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="px-2.5 text-xs" data-testid="echo-on">
            On
          </ToggleGroupItem>
        </ToggleGroup>
        {value.mode === 'on' && (
          <CatalogSelect
            models={models}
            value={value.model}
            onChange={(model) => onChange({ ...value, model })}
            label="Echo canceller"
            defaultLabel={caps.default_model ? `Default (${caps.default_model})` : 'Default canceller'}
            disabled={disabled}
            autoPull={autoPull(info)}
            className="w-56"
            testId="echo-model"
          />
        )}
      </div>
      {value.mode === 'on' && (
        <p className="text-xs leading-4 text-muted-foreground">
          On the server, with what it sent as the reference. The browser already cancels echo on the microphone; this is
          for clients that do not.
          {caps.install && ` SpeexDSP and WebRTC AEC3 need ${caps.install} on the server.`}
        </p>
      )}
      {live && (
        <p className="text-xs text-muted-foreground" data-testid="server-echo">
          Server:{' '}
          {reported
            ? reported.error
              ? `echo cancellation with ${reported.model} failed: ${reported.error}`
              : `cancelling echo with ${reported.model}`
            : 'no echo cancellation'}
        </p>
      )}
    </div>
  )
}
