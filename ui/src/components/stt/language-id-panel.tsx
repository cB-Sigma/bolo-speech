import { FileAudio, Languages, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { CatalogSelect } from '@/components/catalog-select'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { LanguageId, Sample, ServerInfo } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'
import { fmtSeconds, languageName } from '@/lib/format'
import { cn } from '@/lib/utils'

/** `recogniser.reason` of `POST /v1/audio/language`, as `model=auto` would decide (api.md, "Routing"). */
export const ROUTE_REASON: Record<string, string> = {
  route: 'the language_routes setting names it for this language',
  recommended: 'the best installed recogniser that speaks the language',
  default: 'no installed recogniser speaks it, so the default recogniser answers',
  unsure: 'the language was not heard clearly enough, so the default recogniser answers',
  'lid-failed': 'the language identifier could not run, so the default recogniser answers',
}

function pct(p: number): string {
  return p >= 0.995 ? '100 %' : p < 0.005 ? '<1 %' : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)} %`
}

/**
 * "Detect language": a recording heard by a language-identification model on the server (`POST /v1/audio/language`),
 * the most likely languages with their probabilities, and where a transcription with `model=auto` would go.
 */
export function LanguageIdPanel({
  info,
  samples,
  className,
}: {
  info: ServerInfo | undefined
  samples: Sample[]
  className?: string
}) {
  const models = useModels('lid').models
  const [clip, setClip] = useState<Clip | null>(null)
  const [model, setModel] = useState('')
  const [topK, setTopK] = useState('5')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<LanguageId | null>(null)
  const lid = info?.capabilities?.lid
  const defaultModel = lid?.default_model
  const audioSamples = samples.filter((s) => s.kind === 'audio' && s.url && s.id !== 'tone-bursts')

  const pickSample = async (s: Sample) => {
    if (!s.url) return
    try {
      const bytes = await api.bytes(s.url)
      setClip({ blob: new Blob([bytes], { type: 'audio/wav' }), name: `${s.id}.wav`, seconds: s.seconds ?? null })
      setResult(null)
    } catch (err) {
      toast.error('Could not load the sample', { description: describeError(err) })
    }
  }

  const run = async () => {
    if (!clip) return
    setBusy(true)
    setResult(null)
    try {
      setResult(await api.identifyLanguage(clip.blob, clip.name, { model: model || undefined, topK: Number(topK) }))
    } catch (err) {
      toast.error('Could not identify the language', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  const top = result?.languages[0]?.probability ?? 1
  const route = result?.recogniser
  return (
    <Panel
      title="Detect language"
      description="Which language is spoken, and where model=auto would send it"
      className={className}
      data-testid="language-id"
    >
      <div className="grid gap-3">
        <ClipSlot
          label="Recording"
          clip={clip}
          onChange={(c) => {
            setClip(c)
            setResult(null)
          }}
          disabled={busy}
          testId="lid-clip"
        />
        {audioSamples.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-muted-foreground">Samples</span>
            {audioSamples.map((s) => (
              <Button
                key={s.id}
                size="sm"
                variant={clip?.name === `${s.id}.wav` ? 'secondary' : 'ghost'}
                className="border border-border"
                onClick={() => void pickSample(s)}
                disabled={busy}
                data-testid={`lid-sample-${s.id}`}
              >
                <FileAudio /> <span className="num text-xs">{s.id}</span>
              </Button>
            ))}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Language-identification model"
            defaultLabel={defaultModel ? `Server default (${defaultModel})` : 'Server default'}
            disabled={busy}
            autoPull={autoPull(info)}
            extra={(m) => (m.languages.length > 3 ? `${m.languages.length} languages` : null)}
            className="w-64 max-sm:w-full"
            testId="lid-model"
          />
          <Select value={topK} onValueChange={setTopK} disabled={busy}>
            <SelectTrigger className="h-8 w-28" aria-label="How many languages to list" data-testid="lid-topk">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              {['3', '5', '10'].map((k) => (
                <SelectItem key={k} value={k}>
                  Top {k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" onClick={() => void run()} disabled={!clip || busy} data-testid="lid-run">
            {busy ? <Loader2 className="animate-spin" /> : <Languages />} Detect language
          </Button>
        </div>
        {result && (
          <div className="grid gap-2" data-testid="lid-result">
            <p className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-2xl leading-none font-semibold tracking-tight" data-testid="lid-language">
                {result.name ?? languageName(result.language)}
              </span>
              <span className="num text-sm text-muted-foreground">
                {result.language} · {pct(result.probability)}
              </span>
              {result.probability < (lid?.routing?.min_probability ?? 0.5) && (
                <span className="text-xs text-warning-text">not sure: below the routing threshold</span>
              )}
            </p>
            <ol className="grid gap-1" data-testid="lid-topk-list">
              {result.languages.map((l) => (
                <li key={l.language} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-2 text-xs">
                  <span className="truncate">
                    {l.name ?? languageName(l.language)} <span className="num text-muted-foreground">{l.language}</span>
                  </span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-muted" aria-hidden>
                    <span
                      className={cn(
                        'block h-full rounded-full',
                        l === result.languages[0] ? 'bg-chart-1' : 'bg-chart-mute',
                      )}
                      style={{ width: `${Math.max(1, (100 * l.probability) / top)}%` }}
                    />
                  </span>
                  <span className="text-right num">{pct(l.probability)}</span>
                </li>
              ))}
            </ol>
            <p className="text-xs leading-4 text-muted-foreground">
              <span className="num">{result.model}</span>
              {result.heard != null && <> · heard {fmtSeconds(result.heard)}</>}
              {result.duration != null && result.heard != null && result.duration > result.heard && (
                <> of {fmtSeconds(result.duration)}</>
              )}
              . A probability is the model&apos;s own; it always names one of its languages, even for music or silence.
            </p>
            {route && (
              <p className="text-xs leading-4" data-testid="lid-route">
                <span className="text-muted-foreground">model=auto would use </span>
                <span className="num">{route.model ?? 'the default recogniser'}</span>
                {route.language && <span className="text-muted-foreground">, told the language {route.language}</span>}
                <span className="text-muted-foreground">: {ROUTE_REASON[route.reason] ?? route.reason}.</span>
                {route.recommended && (
                  <span className="text-warning-text">
                    {' '}
                    Better for it, not installed: <span className="num">{route.recommended}</span>.
                  </span>
                )}
                {route.detail && <span className="text-muted-foreground"> {route.detail}</span>}
              </p>
            )}
          </div>
        )}
      </div>
    </Panel>
  )
}
