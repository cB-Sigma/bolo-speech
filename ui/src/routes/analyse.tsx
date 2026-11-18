import { AudioLines, Download, FileAudio, Loader2, Smile, Sparkles, Tags } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Panel } from '@/components/panel'
import { BrowserTagsPanel } from '@/components/analyse/browser-tags'
import { TagBars } from '@/components/analyse/tag-bars'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useInfo, useModels, useSamples } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { AudioEmotion, AudioTags, Sample } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'
import { fmtSeconds } from '@/lib/format'

const WINDOWS = ['0', '1', '2', '5', '10']

function WindowSelect({ value, onChange, disabled, testId }: {
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  testId: string
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="h-8 w-40" aria-label="Also window by window" data-testid={testId}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent position="popper" align="start">
        {WINDOWS.map((w) => (
          <SelectItem key={w} value={w}>
            {w === '0' ? 'Whole recording' : `Every ${w} s too`}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function SamplePicker({ samples, clip, onPick, disabled }: {
  samples: Sample[]
  clip: Clip | null
  onPick: (s: Sample) => void
  disabled?: boolean
}) {
  if (!samples.length) return null
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-1 text-xs text-muted-foreground">Samples</span>
      {samples.map((s) => (
        <Button
          key={s.id}
          size="sm"
          variant={clip?.name === `${s.id}.wav` ? 'secondary' : 'ghost'}
          className="border border-border"
          onClick={() => onPick(s)}
          disabled={disabled}
          data-testid={`analyse-sample-${s.id}`}
        >
          <FileAudio /> <span className="num text-xs">{s.id}</span>
        </Button>
      ))}
    </div>
  )
}

/** Sound events heard in the recording (`POST /v1/audio/tags`), whole and window by window. */
function TagsPanel({ clip, autoPullOn, defaultModel }: {
  clip: Clip | null
  autoPullOn: boolean | null
  defaultModel?: string | null
}) {
  const models = useModels('audio-tagging').models
  const [model, setModel] = useState('')
  const [window, setWindow] = useState('0')
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<{ clip: Clip; body: AudioTags } | null>(null)
  const result = answer?.clip === clip ? answer.body : null
  const run = async () => {
    if (!clip) return
    setBusy(true)
    try {
      const body = await api.audioTags(clip.blob, clip.name, {
        model: model || undefined,
        window: Number(window) || undefined,
      })
      setAnswer({ clip, body })
    } catch (err) {
      toast.error('Could not tag the recording', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel
      title="Sound events"
      description="Which of 527 AudioSet sounds the recording holds: speech, music and its instruments, laughter, a dog, a siren..."
      data-testid="analyse-tags"
    >
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Audio-tagging model"
            defaultLabel={defaultModel ? `Server default (${defaultModel})` : 'Server default'}
            disabled={busy}
            autoPull={autoPullOn}
            className="w-64 max-sm:w-full"
            testId="analyse-tags-model"
          />
          <WindowSelect value={window} onChange={setWindow} disabled={busy} testId="analyse-tags-window" />
          <Button size="sm" onClick={() => void run()} disabled={!clip || busy} data-testid="analyse-tags-run">
            {busy ? <Loader2 className="animate-spin" /> : <Tags />} Tag sounds
          </Button>
        </div>
        {result && (
          <div className="grid gap-3" data-testid="analyse-tags-result">
            <TagBars items={result.tags} empty="Nothing above 10 %." />
            {result.segments && (
              <ol className="grid gap-1 text-xs" data-testid="analyse-tags-segments">
                {result.segments.map((s) => (
                  <li key={s.start} className="flex gap-3">
                    <span className="num w-24 shrink-0 text-muted-foreground">
                      {s.start.toFixed(1)}-{s.end.toFixed(1)} s
                    </span>
                    <span className="min-w-0 truncate">
                      {s.tags.map((t) => `${t.label} ${Math.round(t.score * 100)} %`).join(' · ') || '-'}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            <p className="text-xs text-muted-foreground">
              {result.model} · {fmtSeconds(result.duration)} · classes are independent, so the scores do not add up to
              100 %
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}

/** How the speaker sounds (`POST /v1/audio/emotion`). */
function EmotionPanel({ clip, autoPullOn, defaultModel }: {
  clip: Clip | null
  autoPullOn: boolean | null
  defaultModel?: string | null
}) {
  const models = useModels('emotion').models
  const [model, setModel] = useState('')
  const [window, setWindow] = useState('0')
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<{ clip: Clip; body: AudioEmotion } | null>(null)
  const result = answer?.clip === clip ? answer.body : null
  const run = async () => {
    if (!clip) return
    setBusy(true)
    try {
      const body = await api.audioEmotion(clip.blob, clip.name, {
        model: model || undefined,
        window: Number(window) || undefined,
      })
      setAnswer({ clip, body })
    } catch (err) {
      toast.error('Could not recognise the emotion', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel
      title="Emotion"
      description="How the speaker sounds. A guess from the voice, not how anyone feels."
      data-testid="analyse-emotion"
    >
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Emotion model"
            defaultLabel={defaultModel ? `Server default (${defaultModel})` : 'Server default'}
            disabled={busy}
            autoPull={autoPullOn}
            className="w-64 max-sm:w-full"
            testId="analyse-emotion-model"
          />
          <WindowSelect value={window} onChange={setWindow} disabled={busy} testId="analyse-emotion-window" />
          <Button size="sm" onClick={() => void run()} disabled={!clip || busy} data-testid="analyse-emotion-run">
            {busy ? <Loader2 className="animate-spin" /> : <Smile />} Recognise emotion
          </Button>
        </div>
        {result && (
          <div className="grid gap-3" data-testid="analyse-emotion-result">
            <p className="flex items-baseline gap-2">
              <span className="text-2xl leading-none font-semibold tracking-tight" data-testid="analyse-emotion-top">
                {result.emotion ?? '-'}
              </span>
              <span className="num text-sm text-muted-foreground">{Math.round(result.score * 100)} %</span>
            </p>
            <TagBars items={result.emotions.slice(0, 5)} empty="-" />
            {result.segments && (
              <ol className="grid gap-1 text-xs" data-testid="analyse-emotion-segments">
                {result.segments.map((s) => (
                  <li key={s.start} className="flex gap-3">
                    <span className="num w-24 shrink-0 text-muted-foreground">
                      {s.start.toFixed(1)}-{s.end.toFixed(1)} s
                    </span>
                    <span>
                      {s.emotion ?? '-'} {Math.round(s.score * 100)} %
                    </span>
                  </li>
                ))}
              </ol>
            )}
            <p className="text-xs text-muted-foreground">
              {result.model} · {fmtSeconds(result.duration)}
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}

/** Background noise removed (`POST /v1/audio/denoise`), to listen to and download. */
function CleanPanel({ clip, autoPullOn, defaultModel }: {
  clip: Clip | null
  autoPullOn: boolean | null
  defaultModel?: string | null
}) {
  const models = useModels('denoise').models
  const [model, setModel] = useState('')
  const [strength, setStrength] = useState(1)
  const [busy, setBusy] = useState(false)
  const [cleaned, setCleaned] = useState<{ clip: Clip; url: string; model: string | null } | null>(null)
  useEffect(() => () => void (cleaned && URL.revokeObjectURL(cleaned.url)), [cleaned])
  const out = cleaned?.clip === clip ? cleaned : null
  const run = async () => {
    if (!clip) return
    setBusy(true)
    try {
      const r = await api.denoise(clip.blob, clip.name, { model: model || undefined, strength })
      setCleaned({ clip, url: URL.createObjectURL(r.audio), model: r.model })
    } catch (err) {
      toast.error('Could not clean the recording', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel title="Clean up" description="Background noise removed: voice isolation" data-testid="analyse-denoise">
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Denoising model"
            defaultLabel={defaultModel ? `Server default (${defaultModel})` : 'Server default'}
            disabled={busy}
            autoPull={autoPullOn}
            className="w-64 max-sm:w-full"
            testId="analyse-denoise-model"
          />
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            Strength
            <Slider
              className="w-28"
              min={0}
              max={1}
              step={0.05}
              value={[strength]}
              onValueChange={(v) => setStrength(v[0] ?? 1)}
              disabled={busy}
              aria-label="Strength"
            />
            <span className="num w-9">{Math.round(strength * 100)} %</span>
          </label>
          <Button size="sm" onClick={() => void run()} disabled={!clip || busy} data-testid="analyse-denoise-run">
            {busy ? <Loader2 className="animate-spin" /> : <Sparkles />} Clean up
          </Button>
        </div>
        {out && (
          <div className="flex flex-wrap items-center gap-2" data-testid="analyse-denoise-result">
            <audio controls src={out.url} className="h-8 min-w-0 flex-1" aria-label="The cleaned recording" />
            <Button size="sm" variant="outline" asChild>
              <a href={out.url} download={`${clip?.name.replace(/\.[^.]+$/, '') ?? 'audio'}-clean.wav`}>
                <Download /> WAV
              </a>
            </Button>
            {out.model && <span className="text-xs text-muted-foreground">{out.model}</span>}
          </div>
        )}
      </div>
    </Panel>
  )
}

/** Analyse a recording: sound events, emotion and noise removed. */
export default function AnalysePage() {
  const info = useInfo().data
  const samples = useSamples().data?.samples
  const audioSamples = useMemo(
    () => (samples ?? []).filter((s) => s.kind === 'audio' && s.url && s.id !== 'tone-bursts'),
    [samples],
  )
  const [clip, setClip] = useState<Clip | null>(null)
  const caps = info?.capabilities

  const pickSample = async (s: Sample) => {
    if (!s.url) return
    try {
      const bytes = await api.bytes(s.url)
      setClip({ blob: new Blob([bytes], { type: 'audio/wav' }), name: `${s.id}.wav`, seconds: s.seconds ?? null })
    } catch (err) {
      toast.error('Could not load the sample', { description: describeError(err) })
    }
  }

  if (isDemo) {
    return (
      <Page title="Analyse audio">
        <BrowserTagsPanel />
      </Page>
    )
  }
  const on = autoPull(info)
  return (
    <Page title="Analyse audio">
      <div className="grid gap-4">
        <Panel title="Recording" description="Upload a file or record one; every panel below uses it">
          <div className="grid gap-3">
            <ClipSlot label="Recording" clip={clip} onChange={setClip} testId="analyse-clip" />
            <SamplePicker samples={audioSamples} clip={clip} onPick={(s) => void pickSample(s)} />
          </div>
        </Panel>
        <div className="grid gap-4 xl:grid-cols-2">
          <TagsPanel clip={clip} autoPullOn={on} defaultModel={caps?.audio_tagging?.default_model} />
          <EmotionPanel clip={clip} autoPullOn={on} defaultModel={caps?.emotion?.default_model} />
          <CleanPanel clip={clip} autoPullOn={on} defaultModel={caps?.denoise?.default_model} />
        </div>
        {!clip && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <AudioLines className="size-4" /> Pick a recording first.
          </p>
        )}
      </div>
    </Page>
  )
}
