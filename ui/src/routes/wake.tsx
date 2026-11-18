import { Mic, MicOff, Radio } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useInfo, useModels } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { autoPull } from '@/lib/capabilities'
import { Store } from '@/lib/store'
import { idleWake, splitKeywords, WakeListener, type WakeState } from '@/lib/wake'

const idle = new Store<WakeState>(idleWake)

/** Listen to the microphone for wake words (`WS /api/v1/wake/stream`), with any wake-word model. */
export default function WakePage() {
  const info = useInfo()
  const models = useModels('wake-word').models
  const [model, setModel] = useState('')
  const [text, setText] = useState('')
  const [listener, setListener] = useState<WakeListener | null>(null)
  const state = useStore(listener?.store ?? idle)
  const active = state.status === 'starting' || state.status === 'listening'
  const defaultId = info.data?.capabilities?.wake?.default_model ?? null
  const chosen = useMemo(() => models.find((m) => m.id === (model || defaultId)) ?? null, [models, model, defaultId])
  const custom = chosen?.custom_keywords ?? false
  const keywords = custom ? splitKeywords(text) : []

  useEffect(() => () => void listener?.stop(), [listener])

  const start = () => {
    const l = new WakeListener(model, keywords)
    void l.start()
    setListener(l)
  }
  const stop = () => {
    void listener?.stop()
  }

  return (
    <Page title="Wake word">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel
          title="Listen for"
          description="A wake-word model, and the phrases a keyword spotter should hear"
          bodyClassName="grid gap-3"
        >
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Wake-word model"
            defaultLabel={defaultId ? `Server default (${defaultId})` : 'Server default'}
            disabled={active}
            autoPull={autoPull(info.data)}
            extra={(m) => (m.custom_keywords ? 'your phrases' : (m.wake_phrases ?? []).join(', ') || null)}
            testId="wake-model"
          />
          {chosen && (
            <p className="text-xs text-muted-foreground" data-testid="wake-phrases">
              {custom
                ? `Type phrases, one per line or separated by commas. Left empty: ${(chosen.wake_phrases ?? []).join(', ').toLowerCase()}.`
                : `Hears “${(chosen.wake_phrases ?? []).join(', ')}” only.`}
              {chosen.license_note && <span className="block text-warning-text">{chosen.license_note}</span>}
            </p>
          )}
          {custom && (
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={'hey computer\nlights on'}
              aria-label="Phrases to listen for"
              rows={3}
              disabled={active}
              data-testid="wake-keywords"
            />
          )}
          <div>
            {active ? (
              <Button variant="outline" size="sm" onClick={stop} data-testid="wake-stop">
                <MicOff /> Stop
              </Button>
            ) : (
              <Button size="sm" onClick={start} data-testid="wake-start">
                <Mic /> Start listening
              </Button>
            )}
          </div>
        </Panel>
        <Panel
          title="Heard"
          description={
            state.status === 'listening'
              ? `Listening with ${state.model} for ${state.keywords.join(', ')}`
              : state.status === 'starting'
                ? 'Starting…'
                : 'Not listening'
          }
          data-testid="wake-heard"
        >
          {state.error && (
            <Alert variant="destructive" className="mb-3" data-testid="wake-error">
              <AlertTitle>Could not listen</AlertTitle>
              <AlertDescription>{state.error}</AlertDescription>
            </Alert>
          )}
          {state.hits.length ? (
            <ol className="grid gap-1.5" data-testid="wake-hits">
              {state.hits.map((h) => (
                <li key={`${h.at}-${h.time}`} className="flex items-baseline justify-between gap-3 text-[13px]">
                  <span className="font-medium">{h.keyword}</span>
                  <span className="num text-xs text-muted-foreground">
                    at {h.time.toFixed(1)} s{h.score !== undefined ? ` · ${h.score.toFixed(2)}` : ''}
                  </span>
                </li>
              ))}
            </ol>
          ) : (
            <Empty icon={Radio} title={active ? 'Say a phrase' : 'Nothing heard yet'} className="py-8">
              Each time the model hears a phrase it is listed here, newest first.
            </Empty>
          )}
        </Panel>
      </div>
    </Page>
  )
}
