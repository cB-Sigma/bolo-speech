import { Blend, Check, Download, Gauge, Plus, Square, Trash2, TriangleAlert, Volume2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Empty, Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { SavedBlend, SavedVoice } from '@/lib/api/types'
import { download } from '@/lib/bench'
import { fmtSeconds, languageName } from '@/lib/format'
import { cn } from '@/lib/utils'
import { NewVoiceDialog, type VoiceSource } from './new-voice-dialog'
import { CONSENT_LABEL, scoreFor, scoreText, scoreTone, useVoiceMutations, useVoices } from './use-voices'

/**
 * The server's saved voices: play a reference, use it as the voice of the playground, score how much a model's
 * clone sounds like it, save a new one, delete one. `model` is the playground's model; voices can be used and
 * scored only when it clones voices (`canUse`). Saved blends (a mix of a model's own voices) are listed apart:
 * no recording and no consent; they can be used when the model blends voices (`canBlend`).
 */
export function VoiceLibrary({
  model,
  canUse,
  selected,
  onSelect,
  pending,
  onPendingClear,
  canBlend = false,
  selectedBlend = null,
  onSelectBlend,
  disabled,
  className,
}: {
  model: string | null
  canUse: boolean
  canBlend?: boolean
  selectedBlend?: string | null
  onSelectBlend?: (id: string | null) => void
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
  const blends = voices.data?.blends ?? []
  const blending = voices.data?.blending_models ?? []

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

  const del = (v: SavedVoice | SavedBlend) => {
    if (confirming !== v.id) {
      setConfirming(v.id)
      return
    }
    setConfirming(null)
    const blend = 'kind' in v && v.kind === 'blend'
    remove.mutate(v.id, {
      onSuccess: () => {
        if (selected === v.id) onSelect(null)
        if (selectedBlend === v.id) onSelectBlend?.(null)
        toast.success(`Deleted the ${blend ? 'blend' : 'voice'} ${v.name}`)
      },
      onError: (err) =>
        toast.error(`Could not delete the ${blend ? 'blend' : 'voice'}`, { description: describeError(err) }),
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
          : `${list.length} voice${list.length === 1 ? '' : 's'} on the server · for models that clone voices${blends.length ? ` · ${blends.length} blend${blends.length === 1 ? '' : 's'}` : ''}`
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
      {blends.length > 0 && (
        <div className="mt-4 grid gap-1.5" data-testid="blend-list">
          <h3 className="eyebrow">Saved blends</h3>
          {!canBlend && (
            <p className="text-[13px] leading-5 text-muted-foreground" data-testid="blend-library-hint">
              {model ? (
                <>
                  <span className="num">{model}</span> does not blend voices.{' '}
                </>
              ) : null}
              {blending.length
                ? `Choose one that does to use these: ${blending.join(', ')}.`
                : 'Blends are spoken by models that blend voices (Kokoro).'}
            </p>
          )}
          <ul className="grid gap-1.5">
            {blends.map((b) => {
              const active = selectedBlend === b.id
              const other = canBlend && model && b.default_model && b.default_model !== model
              return (
                <li
                  key={b.id}
                  className={cn(
                    'flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border px-2.5 py-1.5',
                    active ? 'border-primary/60 bg-primary/5' : 'border-border',
                  )}
                  data-testid={`blend-${b.id}`}
                >
                  <Blend className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 text-[13px]">
                      <span className="truncate font-medium">{b.name}</span>
                      <span className="num text-xs text-muted-foreground">{b.id}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                      <code className="num break-all">{b.blend}</code>
                      {b.language && <span>{languageName(b.language)}</span>}
                      {b.default_model && (
                        <span className={cn('num', other && 'text-warning-text')}>
                          {other ? `made for ${b.default_model}` : b.default_model}
                        </span>
                      )}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant={active ? 'secondary' : 'outline'}
                    className="h-7"
                    onClick={() => onSelectBlend?.(active ? null : b.id)}
                    disabled={!canBlend || !onSelectBlend || disabled}
                    data-testid={`use-blend-${b.id}`}
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
                    size={confirming === b.id ? 'sm' : 'icon-sm'}
                    variant={confirming === b.id ? 'destructive' : 'ghost'}
                    className={confirming === b.id ? 'h-7' : undefined}
                    onClick={() => del(b)}
                    onBlur={() => setConfirming((c) => (c === b.id ? null : c))}
                    disabled={disabled}
                    aria-label={confirming === b.id ? `Confirm deleting ${b.name}` : `Delete ${b.name}`}
                  >
                    {confirming === b.id ? 'Delete?' : <Trash2 />}
                  </Button>
                </li>
              )
            })}
          </ul>
        </div>
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
