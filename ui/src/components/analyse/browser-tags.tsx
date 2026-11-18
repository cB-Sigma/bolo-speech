import { Loader2, Tags } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { TagBars } from '@/components/analyse/tag-bars'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { useStore } from '@/hooks/use-store'
import { describeError } from '@/hooks/use-api'
import { STT_RATE } from '@/lib/audio/capture'
import { resample } from '@/lib/audio/resample'
import { decodeAudio } from '@/lib/audio/wav'
import { audioEngine } from '@/lib/browser-audio/engine'
import { audioModelsFor } from '@/lib/browser-audio/models'
import { fmtBytes, fmtMs, fmtSeconds } from '@/lib/format'

type Result = {
  model: string
  seconds: number
  ms: number
  tags: { label: string; score: number }[]
  windows: { start: number; end: number; tags: { label: string; score: number }[] }[]
}

const toScores = (ranked: [string, number][]) =>
  ranked.filter(([, p]) => p >= 0.1).map(([label, score]) => ({ label, score }))

/** Sound events in this browser: CED, the tagger the server runs, on WebAssembly. Nothing is uploaded. */
export function BrowserTagsPanel() {
  const models = audioModelsFor('tag')
  const engine = audioEngine('tag')
  const es = useStore(engine.store)
  const [model, setModel] = useState(models[0]?.id ?? '')
  const [window, setWindow] = useState('0')
  const [clip, setClip] = useState<Clip | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const chosen = models.find((m) => m.id === model)

  const run = async () => {
    if (!clip || !chosen) return
    setBusy(true)
    setResult(null)
    try {
      const decoded = await decodeAudio(await clip.blob.arrayBuffer())
      const x =
        decoded.sampleRate === STT_RATE ? decoded.samples : resample(decoded.samples, decoded.sampleRate, STT_RATE)
      await engine.load(chosen.id)
      const r = await engine.tag(x, 10, Number(window) || undefined)
      setResult({
        model: chosen.label,
        seconds: x.length / STT_RATE,
        ms: r.ms,
        tags: toScores(r.ranked),
        windows: r.windows.map((w) => ({ start: w.start, end: w.end, tags: toScores(w.ranked).slice(0, 3) })),
      })
    } catch (err) {
      toast.error('Could not tag the recording', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="Sound events, in this browser"
      description="Which of AudioSet's 527 sounds a recording holds (speech, music and its instruments, laughter, a dog, a siren...), heard by CED on WebAssembly. Nothing is uploaded. A bolo server also recognises emotion, removes noise and answers questions about a recording."
      data-testid="analyse-browser-tags"
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
          testId="analyse-browser-clip"
        />
        <div className="flex flex-wrap items-center gap-2">
          <Select value={model} onValueChange={setModel} disabled={busy}>
            <SelectTrigger className="h-8 w-72" aria-label="Tagging model" data-testid="analyse-browser-model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              {models.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.label} · {fmtBytes(m.bytes)} · {m.license}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={window} onValueChange={setWindow} disabled={busy}>
            <SelectTrigger className="h-8 w-40" aria-label="Also window by window">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              {['0', '1', '2', '5'].map((w) => (
                <SelectItem key={w} value={w}>
                  {w === '0' ? 'Whole recording' : `Every ${w} s too`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" onClick={() => void run()} disabled={!clip || busy} data-testid="analyse-browser-run">
            {busy ? <Loader2 className="animate-spin" /> : <Tags />}
            {busy && es.status === 'loading' ? `Loading ${Math.round(es.progress * 100)} %` : 'Tag sounds'}
          </Button>
        </div>
        {chosen && <p className="text-xs text-muted-foreground">{chosen.blurb}</p>}
        {result && (
          <div className="grid gap-3" data-testid="analyse-browser-result">
            <TagBars items={result.tags} empty="Nothing above 10 %." />
            {result.windows.length > 0 && (
              <ol className="grid gap-1 text-xs">
                {result.windows.map((w) => (
                  <li key={w.start} className="flex gap-3">
                    <span className="w-24 shrink-0 num text-muted-foreground">
                      {w.start.toFixed(1)}-{w.end.toFixed(1)} s
                    </span>
                    <span className="min-w-0 truncate">
                      {w.tags.map((t) => `${t.label} ${Math.round(t.score * 100)} %`).join(' · ') || '-'}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            <p className="text-xs text-muted-foreground">
              {result.model} · {fmtSeconds(result.seconds)} heard in {fmtMs(result.ms)} · classes are independent, so
              the scores do not add up to 100 %
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}
