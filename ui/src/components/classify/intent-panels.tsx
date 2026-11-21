import { Loader2, Target } from 'lucide-react'
import { useState } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { Empty, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useInfo, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { AudioIntent } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'

/** "alarm_set" -> "alarm set"; null -> "not understood". */
export function intentLabel(intent: string | null): string {
  return intent ? intent.replace(/_/g, ' ') : 'not understood'
}

/** The slots as rows, one per filled slot, in the model's order ("time" -> "seven am"). */
export function slotRows(result: AudioIntent): [string, string][] {
  return result.slots.map((s) => [s.type.replace(/_/g, ' '), s.value])
}

/** Spoken intent (`POST /v1/audio/intent`) on the Classify page: what a speaker wants in one utterance, with any
 * intent model. */
export function IntentPanels() {
  const info = useInfo()
  const models = useModels('intent').models
  const [model, setModel] = useState('')
  const [clip, setClip] = useState<Clip | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AudioIntent | null>(null)
  const defaultId = info.data?.capabilities?.intent?.default_model ?? null
  const chosen = models.find((m) => m.id === (model || defaultId)) ?? null

  const run = async () => {
    if (!clip) {
      setError('Upload or record one utterance first')
      return
    }
    setBusy(true)
    setError(null)
    try {
      setResult(await api.audioIntent(clip.blob, { name: clip.name, model: model || undefined }))
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Panel
        title="Spoken intent"
        description="One spoken command or question; the model reads its meaning from the audio, without a transcript"
        bodyClassName="grid gap-3"
      >
        <CatalogSelect
          models={models}
          value={model}
          onChange={setModel}
          label="Intent model"
          defaultLabel={defaultId ? `Server default (${defaultId})` : 'Server default'}
          disabled={busy}
          autoPull={autoPull(info.data)}
          testId="intent-model"
        />
        {chosen?.license_note && <p className="text-xs text-warning-text">{chosen.license_note}</p>}
        <ClipSlot
          label="Recording (one utterance, at most 30 s)"
          clip={clip}
          onChange={setClip}
          disabled={busy}
          testId="intent-clip"
        />
        <div>
          <Button size="sm" onClick={() => void run()} disabled={busy} data-testid="intent-run">
            {busy ? <Loader2 className="animate-spin" /> : <Target />} Understand
          </Button>
        </div>
      </Panel>
      <Panel
        title="Meaning"
        description={result ? `${result.model}, ${result.duration.toFixed(1)} s` : 'The intent and the slots it filled'}
        data-testid="intent-result"
      >
        {error && (
          <Alert variant="destructive" className="mb-3" data-testid="intent-error">
            <AlertTitle>Not understood</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {result ? (
          <div className="grid gap-3 text-[13px]">
            <div>
              <span className="text-muted-foreground">Intent</span>
              <p className="text-lg font-medium" data-testid="intent-name">
                {intentLabel(result.intent)}
              </p>
              {(result.scenario || result.action) && (
                <p className="text-muted-foreground">
                  scenario {result.scenario ?? '-'}, action {result.action ?? '-'}
                </p>
              )}
            </div>
            {result.slots.length > 0 ? (
              <table className="w-full text-left" data-testid="intent-slots">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="font-normal">Slot</th>
                    <th className="font-normal">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {slotRows(result).map(([type, value], i) => (
                    <tr key={i}>
                      <td className="pr-4">{type}</td>
                      <td className="font-medium">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-muted-foreground">No slots filled</p>
            )}
            <p className="font-mono text-xs break-all text-muted-foreground">{result.text}</p>
          </div>
        ) : (
          <Empty icon={Target} title="Nothing understood yet" className="py-8">
            Say something like &ldquo;wake me up at seven tomorrow&rdquo;. The SLURP model knows home-assistant commands
            in English: alarms, calendar, music, weather, lists, email and the like.
          </Empty>
        )}
      </Panel>
    </>
  )
}
