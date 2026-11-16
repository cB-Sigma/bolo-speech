import { Check, Download, Gauge, Plus, Square, Trash2, TriangleAlert, Volume2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Empty, Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { SavedVoice } from '@/lib/api/types'
import { download } from '@/lib/bench'
import { fmtSeconds, languageName } from '@/lib/format'
import { cn } from '@/lib/utils'
import { NewVoiceDialog, type VoiceSource } from './new-voice-dialog'
import { CONSENT_LABEL, scoreFor, scoreText, scoreTone, useVoiceMutations, useVoices } from './use-voices'

/**
 * The server's saved voices: play a reference, use it as the voice of the playground, score how much a model's
 * clone sounds like it, save a new one, delete one. `model` is the playground's model; voices can be used and
 * scored only when it clones voices (`canUse`).
 */
export function VoiceLibrary({
  model,
  canUse,
  selected,
  onSelect,
  pending,
  onPendingClear,
  disabled,
  className,
}: {
  model: string | null
  canUse: boolean
  selected: string | null
  onSelect: (id: string | null) => void
  /** A recording handed over by the clone panel, to save as a voice. */
  pending?: VoiceSource | null
  onPendingClear?: () => void
  disabled?: boolean
  className?: string
}) {
  const voices = useVoices()
  const { remove, score } = useVoiceMutations()
  const [dialog, setDialog] = useState(false)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const audio = useRef<HTMLAudioElement | null>(null)
  const list = voices.data?.voices ?? []
  const cloning = voices.data?.cloning_models ?? []

  useEffect(() => () => audio.current?.pause(), [])

  const play = async (v: SavedVoice) => {
    audio.current?.pause()
    if (playing === v.id) {
      setPlaying(null)
      return
    }
    try {
      const blob = await api.voiceAudio(v.id)
      const url = URL.createObjectURL(blob)
      const a = new Audio(url)
      audio.current = a
      setPlaying(v.id)
      a.onended = () => {
        setPlaying(null)
        URL.revokeObjectURL(url)
      }
      await a.play()
    } catch (err) {
      setPlaying(null)
      toast.error('Could not play the reference', { description: describeError(err) })
    }
  }

  const del = (v: SavedVoice) => {
    if (confirming !== v.id) {
      setConfirming(v.id)
      return
    }
    setConfirming(null)
    remove.mutate(v.id, {
      onSuccess: () => {
        if (selected === v.id) onSelect(null)
        toast.success(`Deleted the voice ${v.name}`)
      },
      onError: (err) => toast.error('Could not delete the voice', { description: describeError(err) }),
    })
  }

  const rate = (v: SavedVoice) => {
    if (!model) return
    score.mutate(
      { id: v.id, model },
      {
        onSuccess: (s) => toast.success(`${v.name} with ${s.model}: ${scoreText(s)}`),
        onError: (err) => toast.error('Could not score the voice', { description: describeError(err) }),
      },
    )
  }

  const exportAll = async () => {
    try {
      const blob = await api.exportVoices()
      download('bolo-voices.zip', new Uint8Array(await blob.arrayBuffer()), 'application/zip')
    } catch (err) {
      toast.error('Could not export the voices', { description: describeError(err) })
    }
  }

  return (
    <Panel
      title="Saved voices"
      description={
        voices.isLoading
          ? 'Loading…'
          : `${list.length} voice${list.length === 1 ? '' : 's'} on the server · for models that clone voices`
      }
      className={className}
      data-testid="voice-library"
      actions={
        <div className="flex items-center gap-1">
          {list.length > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon-sm" variant="ghost" onClick={() => void exportAll()} aria-label="Export all voices">
                  <Download />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export as .zip (bolo voices import reads it)</TooltipContent>
            </Tooltip>
          )}
          <Button size="sm" variant="outline" onClick={() => setDialog(true)} disabled={disabled}>
            <Plus /> New voice
          </Button>
        </div>
      }
    >
      {!canUse && (
        <p className="mb-3 text-[13px] leading-5 text-muted-foreground" data-testid="voice-library-hint">
          {model ? (
            <>
              <span className="num">{model}</span> does not clone voices.{' '}
            </>
          ) : null}
          {cloning.length
            ? `Choose one that does to use these: ${cloning.join(', ')}.`
            : 'Pull a model that clones voices (its catalog entry lists voice-cloning) to use these.'}
        </p>
      )}
      {list.length === 0 ? (
        <Empty title="No saved voices yet" className="py-5">
          Save a recording of a voice you may use: your own, or one whose speaker gave permission.
        </Empty>
      ) : (
        <ul className="grid gap-1.5" data-testid="voice-list">
          {list.map((v) => {
            const s = scoreFor(v, model)
            const tone = scoreTone(s)
            const active = selected === v.id
            return (
              <li
                key={v.id}
                className={cn(
                  'flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border px-2.5 py-1.5',
                  active ? 'border-primary/60 bg-primary/5' : 'border-border',
                )}
                data-testid={`voice-${v.id}`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5 text-[13px]">
                    <span className="truncate font-medium">{v.name}</span>
                    <span className="num text-xs text-muted-foreground">{v.id}</span>
                    {v.warnings.length > 0 && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <TriangleAlert
                            className="size-3.5 text-warning-text"
                            aria-label={`${v.warnings.length} warning(s)`}
                          />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-72">{v.warnings.join(' · ')}</TooltipContent>
                      </Tooltip>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                    <span className="num">{fmtSeconds(v.reference.duration_s)}</span>
                    {v.language && <span>{languageName(v.language)}</span>}
                    <Badge variant="outline" className="h-5 px-1.5 text-[10px]" title={v.consent.statement}>
                      {CONSENT_LABEL[v.consent.basis] ?? v.consent.basis}
                    </Badge>
                    {canUse && model && (
                      <span
                        className={cn(
                          'num',
                          tone === 'good' && 'text-good-text',
                          tone === 'warning' && 'text-warning-text',
                          tone === 'bad' && 'text-destructive',
                        )}
                        data-testid={`voice-score-${v.id}`}
                        title="Speaker similarity of this model's clone to the reference (0.50 and above: what the same real person scores)"
                      >
                        {scoreText(s)}
                      </span>
                    )}
                  </div>
                </div>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  onClick={() => void play(v)}
                  aria-label={playing === v.id ? `Stop ${v.name}` : `Play the reference of ${v.name}`}
                >
                  {playing === v.id ? <Square className="fill-current" /> : <Volume2 />}
                </Button>
                {canUse && model && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        onClick={() => rate(v)}
                        disabled={disabled || score.isPending}
                        aria-label={`Score ${v.name} with ${model}`}
                      >
                        <Gauge />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Speak a test sentence with {model} and measure the likeness</TooltipContent>
                  </Tooltip>
                )}
                <Button
                  size="sm"
                  variant={active ? 'secondary' : 'outline'}
                  className="h-7"
                  onClick={() => onSelect(active ? null : v.id)}
                  disabled={!canUse || disabled}
                  data-testid={`use-voice-${v.id}`}
                >
                  {active ? (
                    <>
                      <Check /> In use
                    </>
                  ) : (
                    'Use'
                  )}
                </Button>
                <Button
                  size={confirming === v.id ? 'sm' : 'icon-sm'}
                  variant={confirming === v.id ? 'destructive' : 'ghost'}
                  className={confirming === v.id ? 'h-7' : undefined}
                  onClick={() => del(v)}
                  onBlur={() => setConfirming((c) => (c === v.id ? null : c))}
                  disabled={disabled}
                  aria-label={confirming === v.id ? `Confirm deleting ${v.name}` : `Delete ${v.name}`}
                >
                  {confirming === v.id ? 'Delete?' : <Trash2 />}
                </Button>
              </li>
            )
          })}
        </ul>
      )}
      {(dialog || pending != null) && (
        <NewVoiceDialog
          open
          onOpenChange={(o) => {
            setDialog(o)
            if (!o) onPendingClear?.()
          }}
          initial={pending}
          scoreModel={canUse ? model : null}
          onCreated={(v) => canUse && onSelect(v.id)}
        />
      )}
    </Panel>
  )
}
