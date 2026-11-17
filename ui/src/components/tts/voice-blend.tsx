import { useQueryClient } from '@tanstack/react-query'
import { BookmarkPlus, Plus, X } from 'lucide-react'
import { useId, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { SavedBlend, Voice } from '@/lib/api/types'
import { blendString, MAX_BLEND_PARTS, normalised, type BlendPart } from '@/lib/blend'

/** Two voices at 0.7 / 0.3 to start from: the first two of the model, else the one it has. */
export function startingBlend(voices: Voice[]): BlendPart[] {
  const [a, b] = voices
  if (!a) return []
  return b
    ? [
        { voice: a.id, weight: 0.7 },
        { voice: b.id, weight: 0.3 },
      ]
    : [{ voice: a.id, weight: 1 }]
}

/**
 * A blend of a model's own voices (Kokoro): pick voices and weights; the playground speaks it instead of the voice
 * picked above. With `canSave` (a server with a voice library) it can be kept under a name.
 */
export function VoiceBlendPanel({
  model,
  voices,
  enabled,
  onEnabledChange,
  parts,
  onPartsChange,
  canSave,
  onSaved,
  disabled,
  className,
}: {
  model: string
  voices: Voice[]
  enabled: boolean
  onEnabledChange: (on: boolean) => void
  parts: BlendPart[]
  onPartsChange: (parts: BlendPart[]) => void
  canSave?: boolean
  onSaved?: (b: SavedBlend) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const qc = useQueryClient()
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const spelled = blendString(parts)
  const shares = normalised(parts)
  const voiceName = (v: string) => voices.find((x) => x.id === v)?.name ?? v
  const duplicate = new Set(parts.map((p) => p.voice)).size < parts.length

  const setPart = (i: number, patch: Partial<BlendPart>) =>
    onPartsChange(parts.map((p, j) => (j === i ? { ...p, ...patch } : p)))
  const add = () => {
    const unused = voices.find((v) => !parts.some((p) => p.voice === v.id)) ?? voices[0]
    if (unused) onPartsChange([...parts, { voice: unused.id, weight: 0.5 }])
  }

  const save = async () => {
    if (!spelled || !name.trim()) return
    setSaving(true)
    try {
      const b = await api.createBlend({ name: name.trim(), blend: spelled, model })
      toast.success(`Saved the blend ${b.name}`, { description: b.blend })
      setName('')
      void qc.invalidateQueries({ queryKey: ['voices'] })
      onSaved?.(b)
    } catch (err) {
      toast.error('Could not save the blend', { description: describeError(err) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Panel
      title="Blend voices"
      description="This model can mix its own voices into a new one"
      className={className}
      data-testid="voice-blend"
      actions={
        <Switch
          checked={enabled}
          onCheckedChange={onEnabledChange}
          disabled={disabled || voices.length === 0}
          size="sm"
          aria-label="Speak a blend of voices instead of one voice"
          data-testid="blend-toggle"
        />
      }
    >
      <div className="grid gap-3">
        <p className="text-[13px] leading-5 text-muted-foreground">
          The voices&apos; styles are mixed in proportion to their weights (scaled to add up to 1); the text is read in
          the language of the heaviest voice. The first request with a new blend takes a few seconds longer.
        </p>
        <ul className="grid gap-1.5" aria-label="Voices in the blend">
          {parts.map((p, i) => {
            const share = shares.find((s) => s.voice === p.voice)?.weight ?? 0
            return (
              <li key={i} className="flex flex-wrap items-center gap-2" data-testid={`blend-part-${i}`}>
                <Select value={p.voice} onValueChange={(v) => setPart(i, { voice: v })} disabled={disabled || !enabled}>
                  <SelectTrigger className="h-8 w-48" aria-label={`Voice ${i + 1} of the blend`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" className="max-h-72">
                    {voices.map((v) => (
                      <SelectItem key={v.id} value={v.id}>
                        {v.name}
                        <span className="ml-2 num text-xs text-muted-foreground">
                          {v.language}
                          {v.gender ? ` · ${v.gender}` : ''}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  type="number"
                  min={0}
                  step={0.05}
                  value={Number.isFinite(p.weight) ? p.weight : ''}
                  onChange={(e) => setPart(i, { weight: e.target.value === '' ? NaN : Number(e.target.value) })}
                  disabled={disabled || !enabled}
                  className="h-8 w-20 num"
                  aria-label={`Weight of ${voiceName(p.voice)}`}
                />
                <span className="w-20 num text-xs text-muted-foreground" title="Its share once the weights are scaled">
                  {(share * 100).toFixed(0)} %
                </span>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  onClick={() => onPartsChange(parts.filter((_, j) => j !== i))}
                  disabled={disabled || !enabled || parts.length <= 1}
                  aria-label={`Remove ${voiceName(p.voice)} from the blend`}
                >
                  <X />
                </Button>
              </li>
            )
          })}
        </ul>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={add}
            disabled={disabled || !enabled || parts.length >= MAX_BLEND_PARTS || voices.length === 0}
            data-testid="blend-add"
          >
            <Plus /> Add a voice
          </Button>
          <code
            className="min-w-0 rounded-md border border-border bg-muted/40 px-2 py-1 num text-xs break-all"
            data-testid="blend-string"
          >
            {spelled || 'no voices with a weight above 0'}
          </code>
        </div>
        {parts.length >= MAX_BLEND_PARTS && (
          <p className="text-xs text-muted-foreground">A blend mixes at most {MAX_BLEND_PARTS} voices.</p>
        )}
        {duplicate && (
          <p className="text-xs text-warning-text">A voice named twice counts once, with its weights added up.</p>
        )}
        {canSave && (
          <div className="flex flex-wrap items-end gap-2 border-t border-border pt-3">
            <label className="grid min-w-0 flex-1 gap-1 text-xs" htmlFor={`${id}-name`}>
              <span className="text-muted-foreground">Keep it in the voice library as</span>
              <Input
                id={`${id}-name`}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Warm mix"
                maxLength={100}
                disabled={disabled || !enabled}
                className="h-8"
              />
            </label>
            <Button
              size="sm"
              variant="outline"
              onClick={() => void save()}
              disabled={disabled || !enabled || !spelled || !name.trim() || saving}
              data-testid="blend-save"
            >
              <BookmarkPlus /> Save blend
            </Button>
          </div>
        )}
      </div>
    </Panel>
  )
}
