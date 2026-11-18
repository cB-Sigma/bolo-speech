import { AudioLines, Download, Info } from 'lucide-react'
import { useState } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { CopyCommand } from '@/components/component-install'
import { Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useCatalog, useModelActions, useModels } from '@/hooks/use-api'
import type { CatalogModel, ServerInfo } from '@/lib/api/types'
import { autoPull, isCloud, isFullDuplex, licenceTags, remoteCodeCommand } from '@/lib/capabilities'
import type { PipelineId } from '@/lib/capabilities'
import { cn } from '@/lib/utils'

const SETTING: Record<'audio-llm' | 'speech-to-speech', string> = {
  'audio-llm': 'audio_llm_model',
  'speech-to-speech': 's2s_model',
}

function kindOf(m: CatalogModel): string {
  if (m.task === 'audio-llm') return 'hears; bolo speaks the answer'
  return isFullDuplex(m) ? 'full-duplex' : 'turn-based'
}

/** What a full-duplex model does differently, in the console's words. */
export function FullDuplexNote({ className }: { className?: string }) {
  return (
    <p className={cn('text-xs leading-4 text-muted-foreground', className)} data-testid="full-duplex-note">
      <span className="font-medium text-foreground">Full duplex:</span> the model hears you and speaks at the same time
      and decides itself when to answer. Talking over it does not cut it off, as it does with the other pipelines; press{' '}
      <span className="font-medium text-foreground">Stop the answer</span> to stop what it is saying (it keeps
      listening). Use headphones or the browser’s echo cancellation: it hears whatever the microphone picks up.
    </p>
  )
}

/**
 * The model behind the audio-LLM or speech-to-speech pipeline: every catalog model of that kind with its licence and
 * labels, the one this server runs marked. The model is a server setting (`audio_llm_model`, `s2s_model`), not a
 * session field, so choosing another one here says how to switch the server to it (and pulls it).
 */
export function PipelineModelPanel({
  pipeline,
  info,
  className,
}: {
  pipeline: PipelineId
  info: ServerInfo | undefined
  className?: string
}) {
  const task = pipeline === 'audio-llm' ? 'audio-llm' : 'speech-to-speech'
  const ofTask = useModels(task).models
  const catalog = useCatalog().data?.models
  const actions = useModelActions()
  const entry = info?.capabilities?.realtime?.pipelines?.[pipeline]
  const configured = entry?.model ?? null
  // the server's model may be listed under another task (s2s_model defaults to LFM3-Audio, a text-to-speech entry):
  // it is still the one sessions use, so the list shows it rather than an empty choice
  const extra =
    configured && !ofTask.some((x) => x.id === configured) ? catalog?.find((x) => x.id === configured) : null
  const models = extra ? [extra, ...ofTask] : ofTask
  const [chosen, setChosen] = useState<string>('')
  if (pipeline === 'cascade') return null
  const shown = chosen || configured || ''
  const m = models.find((x) => x.id === shown) ?? null
  const other = Boolean(chosen && chosen !== configured)
  const setting = SETTING[task]
  const fullDuplex = m ? isFullDuplex(m) : Boolean(entry?.full_duplex)
  const tags = m ? licenceTags(m) : []
  return (
    <Panel
      title={pipeline === 'audio-llm' ? 'Audio-in model' : 'Speech-to-speech model'}
      description={
        pipeline === 'audio-llm'
          ? 'Hears your turn itself; bolo speaks its answer'
          : 'Hears you and answers in its own voice: turn-based, or full-duplex'
      }
      className={className}
      data-testid="pipeline-model"
    >
      <div className="grid gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={shown}
            onChange={setChosen}
            label={pipeline === 'audio-llm' ? 'Audio-in model' : 'Speech-to-speech model'}
            defaultLabel={null}
            autoPull={autoPull(info)}
            extra={kindOf}
            className="w-72 max-sm:w-full"
            testId="pipeline-model-select"
          />
          {configured && !other && (
            <Badge variant="outline" className="text-[11px]" data-testid="pipeline-model-running">
              this server runs it
            </Badge>
          )}
          {m && fullDuplex && (
            <Badge variant="outline" className="gap-1 text-[11px]" data-testid="full-duplex-badge">
              <AudioLines className="size-3" aria-hidden /> full duplex
            </Badge>
          )}
        </div>
        {!configured && (
          <p className="text-muted-foreground">
            No model is set: the server needs <span className="num">{setting}</span>
            {pipeline === 'audio-llm' ? ' (or audio_llm_base_url)' : ' (or s2s_server_url)'}.
          </p>
        )}
        {m && (
          <p className="leading-4 text-muted-foreground" data-testid="pipeline-model-about">
            <span className="text-foreground">{m.license}</span>
            {tags.length > 0 && <span className="text-warning-text"> · {tags.join(' · ')}</span>}
            {m.license_note && <span> · {m.license_note}</span>}
            {m.requires?.length ? ` · runs on the component ${m.requires.join(', ')}` : ''}
          </p>
        )}
        {m?.remote_code && !m.remote_code.allowed && (
          <div className="grid gap-1.5" data-testid="pipeline-model-remote-code">
            <p className="text-warning-text">
              Its model code is remote code: it will not load until you opt in for it (read the files first).
            </p>
            <CopyCommand command={remoteCodeCommand(m.id)} label="Opt-in command" />
          </div>
        )}
        {fullDuplex && <FullDuplexNote />}
        {other && m && (
          <div
            className="grid gap-2 rounded-lg border border-border bg-muted/30 p-3"
            data-testid="pipeline-model-switch"
          >
            <p className="flex items-start gap-1.5 leading-4">
              <Info className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-hidden />
              <span>
                Sessions use the server’s model
                {configured ? (
                  <>
                    , <span className="num">{configured}</span>
                  </>
                ) : (
                  ''
                )}
                . The model is a server setting, not chosen per conversation: to talk to{' '}
                <span className="num">{m.id}</span>, set it and restart bolo.
              </span>
            </p>
            <CopyCommand command={`${setting} = "${m.id}"`} label="Setting for bolo.toml" />
            <p className="text-muted-foreground">
              or{' '}
              <span className="num">
                BOLO_{setting.toUpperCase()}={m.id}
              </span>{' '}
              in the environment
              {pipeline === 'speech-to-speech' ? ', with realtime_pipeline = "speech-to-speech" for a default' : ''}.
            </p>
            {!m.installed && !isCloud(m) && (
              <Button
                size="sm"
                variant="outline"
                className="justify-self-start"
                disabled={Boolean(m.pulling) || actions.pull.isPending}
                onClick={() => actions.pull.mutate(m.id)}
                data-testid="pipeline-model-pull"
              >
                <Download /> {m.pulling ? `Pulling ${Math.round(100 * (m.pulling.progress ?? 0))} %` : 'Pull it now'}
              </Button>
            )}
          </div>
        )}
      </div>
    </Panel>
  )
}
