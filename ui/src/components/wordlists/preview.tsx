import { Eye, Play, Square } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { CatalogModel, WordListPreview } from '@/lib/api/types'
import { OverrideText } from './override-text'

export const MAX_PREVIEW_MODELS = 4

const PRONUNCIATION_LABEL: Record<string, string> = {
  phonemes: 'phonemes',
  markup: 'markup',
  respelling: 'respelling',
}

/** Choose up to four voices to compare. */
export function PreviewModelPicker({
  models,
  value,
  onChange,
}: {
  models: CatalogModel[]
  value: string[]
  onChange: (ids: string[]) => void
}) {
  const id = useId()
  const full = value.length >= MAX_PREVIEW_MODELS
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1.5 text-xs font-medium">
        Voices to compare <span className="font-normal text-muted-foreground">(up to {MAX_PREVIEW_MODELS})</span>
      </legend>
      {models.length === 0 ? (
        <p className="text-xs text-muted-foreground">No text-to-speech model is installed.</p>
      ) : (
        <ul className="flex max-h-40 flex-wrap gap-x-4 gap-y-1.5 overflow-y-auto">
          {models.map((m) => {
            const on = value.includes(m.id)
            return (
              <li key={m.id} className="flex items-center gap-2">
                <Checkbox
                  id={`${id}-${m.id}`}
                  checked={on}
                  disabled={!on && full}
                  onCheckedChange={(v) => onChange(v === true ? [...value, m.id] : value.filter((x) => x !== m.id))}
                />
                <Label htmlFor={`${id}-${m.id}`} className="num text-xs font-normal">
                  {m.id}
                </Label>
              </li>
            )
          })}
        </ul>
      )}
    </fieldset>
  )
}

/**
 * What each chosen voice would receive with this word list: the prepared text (overrides as chips),
 * the phonemes for phoneme voices, the entries applied, and a play button that synthesises it.
 */
export function WordListPreviewPane({
  name,
  models,
  chosen,
  onChosenChange,
  className,
}: {
  name: string
  models: CatalogModel[]
  chosen: string[]
  onChosenChange: (ids: string[]) => void
  className?: string
}) {
  const id = useId()
  const [text, setText] = useState('Call bolo support on MH12 2G lines.')
  const [ssml, setSsml] = useState(false)
  const [busy, setBusy] = useState(false)
  const [previews, setPreviews] = useState<WordListPreview[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const audio = useRef<HTMLAudioElement | null>(null)

  useEffect(
    () => () => {
      audio.current?.pause()
    },
    [],
  )

  const run = async () => {
    if (!text.trim() || chosen.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.previewWordList(name, {
        text,
        models: chosen.slice(0, MAX_PREVIEW_MODELS),
        input_format: ssml ? 'ssml' : 'text',
      })
      setPreviews(r.previews)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const play = async (p: WordListPreview) => {
    audio.current?.pause()
    if (playing === p.model) {
      setPlaying(null)
      return
    }
    setPlaying(p.model)
    try {
      const res = await api.speech({
        model: p.model,
        input: text,
        voice: p.voice ?? undefined,
        response_format: 'wav',
        input_format: ssml ? 'ssml' : 'text',
        wordlists: [name],
      })
      const url = URL.createObjectURL(await res.blob())
      const a = new Audio(url)
      audio.current = a
      a.onended = () => {
        URL.revokeObjectURL(url)
        setPlaying(null)
      }
      await a.play()
    } catch (err) {
      setPlaying(null)
      toast.error(`Could not play ${p.model}`, { description: describeError(err) })
    }
  }

  return (
    <Panel
      title="Preview"
      description="What each voice would receive with this list. Nothing is synthesised until you press play."
      className={className}
      data-testid="wordlist-preview"
    >
      <div className="grid gap-3">
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-text`} className="text-xs">
            Text
          </Label>
          <Textarea
            id={`${id}-text`}
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="min-h-20 text-[14px]"
            placeholder={ssml ? '<speak>…</speak>' : 'Type a sentence with the words of the list'}
          />
        </div>
        <PreviewModelPicker models={models} value={chosen} onChange={onChosenChange} />
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs">
            <Switch checked={ssml} onCheckedChange={setSsml} size="sm" aria-label="Text is SSML" />
            SSML
          </label>
          <Button
            size="sm"
            onClick={() => void run()}
            disabled={busy || !text.trim() || chosen.length === 0}
            data-testid="preview-run"
          >
            <Eye /> {busy ? 'Preparing…' : 'Preview'}
          </Button>
        </div>
        {error && (
          <p role="alert" className="rounded-lg border border-critical/30 bg-critical/10 p-3 text-[13px] leading-5">
            {error}
          </p>
        )}
        {previews && (
          <ul className="grid gap-3" data-testid="preview-results">
            {previews.map((p) => (
              <li key={p.model} className="min-w-0 rounded-lg border border-border p-3">
                <div className="mb-1.5 flex flex-wrap items-center gap-2">
                  <span className="num text-[13px] font-medium">{p.model}</span>
                  {p.voice && <span className="text-xs text-muted-foreground">{p.voice}</span>}
                  <Badge variant="outline" className="font-normal" title="How word-list entries reach this voice">
                    {PRONUNCIATION_LABEL[p.text_input.pronunciation] ?? p.text_input.pronunciation}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    reads {p.text_input.reads} · normaliser {p.normalizer}
                  </span>
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    className="ml-auto"
                    onClick={() => void play(p)}
                    aria-label={playing === p.model ? `Stop ${p.model}` : `Play ${p.model}`}
                  >
                    {playing === p.model ? <Square className="fill-current" /> : <Play className="fill-current" />}
                  </Button>
                </div>
                <OverrideText text={p.text} />
                {p.phonemes && (
                  <pre className="mt-1.5 overflow-x-auto rounded-md bg-muted/50 px-2 py-1 num text-[12px] leading-5 whitespace-pre-wrap">
                    {p.phonemes}
                  </pre>
                )}
                {p.applied.length > 0 ? (
                  <ul className="mt-1.5 flex flex-wrap gap-1.5 text-xs" aria-label="Entries applied">
                    {p.applied.map((a, i) => (
                      <li key={i} className="rounded border border-border px-1.5 py-0.5">
                        {a.text} → <span className="num">{a.spoken ?? (a.ipa ? `/${a.ipa}/` : '?')}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-1.5 text-xs text-muted-foreground">No entry of the list matched.</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  )
}
