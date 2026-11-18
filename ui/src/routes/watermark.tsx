import { ShieldCheck, ShieldQuestion, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { describeError, useModels } from '@/hooks/use-api'
import { api, type WatermarkDetection } from '@/lib/api/client'
import { fmtSeconds } from '@/lib/format'

const DEFAULT = '__default__'

/** Drop a clip in to see whether it carries a watermark, where, and the payload. */
export default function WatermarkPage() {
  const { models } = useModels('watermark')
  const [model, setModel] = useState(DEFAULT)
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState<string | null>(null)
  const [result, setResult] = useState<WatermarkDetection | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const check = async (file: File) => {
    setBusy(true)
    setName(file.name)
    setResult(null)
    try {
      setResult(await api.detectWatermark(file, model === DEFAULT ? undefined : model))
    } catch (err) {
      toast.error('The clip could not be checked', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  const toolbar = (
    <Select value={model} onValueChange={setModel} disabled={busy}>
      <SelectTrigger className="h-8 w-60" aria-label="Watermark model" data-testid="watermark-model">
        <SelectValue />
      </SelectTrigger>
      <SelectContent position="popper" align="start">
        <SelectItem value={DEFAULT}>The server's watermarker</SelectItem>
        <SelectItem value="all">Every installed watermarker</SelectItem>
        {models.map((m) => (
          <SelectItem key={m.id} value={m.id}>
            {m.id}
            <span className="ml-2 text-xs text-muted-foreground">{m.license}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const duration = result?.duration ?? 0
  return (
    <Page title="Watermark" toolbar={toolbar}>
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Check a clip" description="Was it made by a machine that marks its speech?">
          <div
            className={`flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-8 text-center ${
              dragging ? 'border-primary bg-muted' : 'border-border'
            }`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const f = e.dataTransfer.files[0]
              if (f) void check(f)
            }}
            data-testid="watermark-drop"
          >
            <p className="text-[13px] text-muted-foreground">Drop a recording here (WAV, MP3, Opus, FLAC, ...)</p>
            <Button variant="outline" size="sm" onClick={() => input.current?.click()} disabled={busy}>
              <Upload /> {busy ? 'Checking…' : 'Choose a file'}
            </Button>
            <input
              ref={input}
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.ogg,.opus,.m4a,.webm"
              className="sr-only"
              tabIndex={-1}
              aria-label="Recording to check"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void check(f)
                e.target.value = ''
              }}
            />
          </div>
        </Panel>
        <Panel title="Result">
          {!result ? (
            <Empty icon={ShieldQuestion} title={busy ? 'Checking…' : 'No clip yet'} />
          ) : (
            <div className="grid gap-3" data-testid="watermark-result">
              <div className="flex items-center gap-2">
                {result.marked ? (
                  <ShieldCheck className="size-5 text-good-text" aria-hidden />
                ) : (
                  <ShieldQuestion className="size-5 text-muted-foreground" aria-hidden />
                )}
                <span className="text-[15px]">
                  {result.marked ? 'Watermarked' : 'No watermark found'}
                  <span className="ml-2 text-xs text-muted-foreground">{name}</span>
                </span>
              </div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                <dt className="text-muted-foreground">Watermarker</dt>
                <dd className="text-right num">{result.model}</dd>
                <dt className="text-muted-foreground">Share of the clip marked</dt>
                <dd className="text-right num">{Math.round(result.score * 100)} %</dd>
                {result.marked && result.payload && (
                  <>
                    <dt className="text-muted-foreground">Payload</dt>
                    <dd className="text-right num">
                      {result.payload}
                      {result.payload_is_this_servers ? ' (this server)' : ''}
                    </dd>
                  </>
                )}
              </dl>
              {duration > 0 && result.segments.length > 0 && (
                <div>
                  <div className="relative h-3 w-full rounded bg-muted" aria-label="Where the mark is">
                    {result.segments.map((g) => (
                      <div
                        key={g.start}
                        className="absolute top-0 h-3 rounded bg-primary"
                        style={{
                          left: `${(g.start / duration) * 100}%`,
                          width: `${((g.end - g.start) / duration) * 100}%`,
                        }}
                        title={`${fmtSeconds(g.start)} – ${fmtSeconds(g.end)}`}
                      />
                    ))}
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Marked: {result.segments.map((g) => `${fmtSeconds(g.start)}–${fmtSeconds(g.end)}`).join(', ')}
                  </p>
                </div>
              )}
              {result.results && result.results.length > 1 && (
                <ul className="grid gap-1 text-xs text-muted-foreground">
                  {result.results.map((r) => (
                    <li key={r.model} className="flex justify-between">
                      <span className="num">{r.model}</span>
                      <span>
                        {r.marked ? 'marked' : 'not marked'} · {Math.round(r.score * 100)} %
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          <p className="mt-4 text-xs leading-5 text-muted-foreground">
            A mark survives cutting, MP3 and Opus, and resampling. No mark does not prove a clip is real: speech from
            other sources carries none.
          </p>
        </Panel>
      </div>
    </Page>
  )
}
