import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Fingerprint, Loader2, Play, Plus, Trash2, UserPlus, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useInfo, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { EnrolledSpeaker, SpeakerIdentification, SpeakerVerification } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'
import { cn } from '@/lib/utils'

const ANYONE = '__anyone'

function Score({ value, threshold }: { value: number | null; threshold: number }) {
  if (value == null) return <span className="text-muted-foreground">-</span>
  return (
    <span className={cn('num', value >= threshold ? 'text-good-text' : 'text-muted-foreground')}>
      {value.toFixed(3)}
    </span>
  )
}

/** An enrolment recording, fetched (with the API key) when asked for. */
function ClipPlayer({ path, label }: { path: string; label: string }) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url])
  if (url) return <audio controls autoPlay src={url} className="h-8 w-56" aria-label={label} />
  const load = async () => {
    try {
      setUrl(URL.createObjectURL(new Blob([await api.bytes(path)], { type: 'audio/wav' })))
    } catch (err) {
      toast.error('Could not load the recording', { description: describeError(err) })
    }
  }
  return (
    <Button size="sm" variant="outline" onClick={() => void load()} aria-label={`Play ${label}`}>
      <Play /> <span className="num text-xs">{label.split(': ').pop()}</span>
    </Button>
  )
}

/** Enrol a speaker from one or more recordings of them alone. */
function EnrolPanel({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState('')
  const [clips, setClips] = useState<(Clip | null)[]>([null])
  const [busy, setBusy] = useState(false)
  const ready = clips.filter((c): c is Clip => c != null)
  const enrol = async () => {
    setBusy(true)
    try {
      const s = await api.enrolSpeaker(name.trim(), ready)
      toast.success(`Enrolled ${s.name}`, { description: `${s.clips.length} recording(s), ${s.seconds.toFixed(1)} s` })
      setName('')
      setClips([null])
      onDone()
    } catch (err) {
      toast.error('Could not enrol the speaker', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel
      title="Enrol a speaker"
      description="A name and recordings of that person alone, 1 to 30 s each. Two or three recordings make a steadier voiceprint."
      data-testid="speakers-enrol"
    >
      <div className="grid gap-3">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name"
          aria-label="Name"
          className="h-8 max-w-sm"
          data-testid="speakers-enrol-name"
        />
        {clips.map((c, i) => (
          <ClipSlot
            key={i}
            label={`Recording ${i + 1}`}
            clip={c}
            onChange={(next) => setClips((all) => all.map((x, j) => (j === i ? next : x)))}
            disabled={busy}
            testId={`speakers-enrol-clip-${i}`}
          />
        ))}
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setClips((all) => [...all, null])}
            disabled={busy || clips.length >= 5}
          >
            <Plus /> Another recording
          </Button>
          <Button
            size="sm"
            onClick={() => void enrol()}
            disabled={busy || !name.trim() || ready.length === 0}
            data-testid="speakers-enrol-run"
          >
            {busy ? <Loader2 className="animate-spin" /> : <UserPlus />} Enrol
          </Button>
        </div>
      </div>
    </Panel>
  )
}

/** Verify a recording against one speaker, or identify it among all of them. */
function CheckPanel({ speakers, thresholds }: { speakers: EnrolledSpeaker[]; thresholds: Record<string, number> }) {
  const info = useInfo().data
  const embedders = useModels('speaker-embedding').models
  const [clip, setClip] = useState<Clip | null>(null)
  const [who, setWho] = useState(ANYONE)
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [verified, setVerified] = useState<SpeakerVerification | null>(null)
  const [identified, setIdentified] = useState<SpeakerIdentification | null>(null)
  const run = async () => {
    if (!clip) return
    setBusy(true)
    setVerified(null)
    setIdentified(null)
    try {
      if (who === ANYONE) setIdentified(await api.identifySpeaker(clip.blob, clip.name, { model: model || undefined }))
      else setVerified(await api.verifySpeaker(clip.blob, clip.name, who, { model: model || undefined }))
    } catch (err) {
      toast.error('Could not compare the voice', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  const result = verified ?? identified
  return (
    <Panel
      title="Who is speaking?"
      description="Identify a recording among the enrolled speakers, or verify it against one of them"
      data-testid="speakers-check"
    >
      <div className="grid gap-3">
        <ClipSlot label="Recording" clip={clip} onChange={setClip} disabled={busy} testId="speakers-check-clip" />
        <div className="flex flex-wrap items-center gap-2">
          <Select value={who} onValueChange={setWho} disabled={busy}>
            <SelectTrigger className="h-8 w-56" aria-label="Compare with" data-testid="speakers-check-who">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" align="start">
              <SelectItem value={ANYONE}>Anyone enrolled (identify)</SelectItem>
              {speakers.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  Is it {s.name}? (verify)
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <CatalogSelect
            models={embedders}
            value={model}
            onChange={setModel}
            label="Speaker embedder"
            defaultLabel="Server default"
            disabled={busy}
            autoPull={autoPull(info)}
            extra={(m) => (thresholds[m.id] != null ? `threshold ${thresholds[m.id]}` : null)}
            className="w-64 max-sm:w-full"
            testId="speakers-check-model"
          />
          <Button
            size="sm"
            onClick={() => void run()}
            disabled={!clip || busy || speakers.length === 0}
            data-testid="speakers-check-run"
          >
            {busy ? <Loader2 className="animate-spin" /> : <Fingerprint />} {who === ANYONE ? 'Identify' : 'Verify'}
          </Button>
        </div>
        {result && (
          <div className="grid gap-2" data-testid="speakers-check-result">
            {verified && (
              <p className="text-lg font-semibold tracking-tight" data-testid="speakers-verdict">
                {verified.match ? `Same speaker as ${verified.speaker.name}` : `Not ${verified.speaker.name}`}
              </p>
            )}
            {identified && (
              <p className="text-lg font-semibold tracking-tight" data-testid="speakers-verdict">
                {identified.speaker ? identified.speaker.name : 'Nobody enrolled'}
              </p>
            )}
            {identified && identified.candidates.length > 0 && (
              <ul className="grid gap-1 text-sm">
                {identified.candidates.map((c) => (
                  <li key={c.id} className="flex gap-3">
                    <Score value={c.similarity} threshold={identified.threshold} />
                    <span>{c.name}</span>
                  </li>
                ))}
              </ul>
            )}
            <p className="text-xs text-muted-foreground">
              {verified && (
                <>
                  Similarity <Score value={verified.similarity} threshold={verified.threshold} /> ·{' '}
                </>
              )}
              threshold {result.threshold} ({result.calibrated ? 'measured for' : 'not measured for'}{' '}
              {result.embedding_model}). A voiceprint is not proof of identity: noise, telephone audio, short clips
              and imitated or synthesised voices move the score.
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}

/** Enrolled speakers: enrol people from recordings, then verify or identify who speaks (`/v1/audio/speakers`). */
export default function SpeakersPage() {
  const qc = useQueryClient()
  const list = useQuery({ queryKey: ['speakers'], queryFn: () => api.speakers() })
  const refresh = () => void qc.invalidateQueries({ queryKey: ['speakers'] })
  const speakers = list.data?.data ?? []

  const remove = async (s: EnrolledSpeaker) => {
    if (!window.confirm(`Delete ${s.name} and their recordings?`)) return
    try {
      await api.deleteSpeaker(s.id)
      refresh()
    } catch (err) {
      toast.error('Could not delete the speaker', { description: describeError(err) })
    }
  }
  const removeClip = async (s: EnrolledSpeaker, clip: string) => {
    try {
      await api.deleteSpeakerClip(s.id, clip)
      refresh()
    } catch (err) {
      toast.error('Could not remove the recording', { description: describeError(err) })
    }
  }

  return (
    <Page title="Speakers">
      <div className="grid gap-4 xl:grid-cols-2">
        <EnrolPanel onDone={refresh} />
        <CheckPanel speakers={speakers} thresholds={list.data?.thresholds ?? {}} />
        <Panel
          title="Enrolled speakers"
          description={
            list.data
              ? `Compared with ${list.data.embedding_model} by default: same speaker at a similarity of ${list.data.threshold} or more${list.data.calibrated ? ' (measured)' : ''}`
              : undefined
          }
          className="xl:col-span-2"
          data-testid="speakers-list"
        >
          {list.error ? (
            <ErrorState error={describeError(list.error)} onRetry={() => void list.refetch()} />
          ) : speakers.length === 0 ? (
            <Empty icon={Users} title="Nobody enrolled yet">
              Enrol someone from a recording of them alone.
            </Empty>
          ) : (
            <ul className="grid gap-3">
              {speakers.map((s) => (
                <li key={s.id} className="grid gap-2 rounded-md border border-border p-3" data-testid={`speaker-${s.id}`}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{s.name}</span>
                    <span className="num text-xs text-muted-foreground">
                      {s.id} · {s.clips.length} recording(s) · {s.seconds.toFixed(1)} s
                    </span>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="ml-auto"
                      onClick={() => void remove(s)}
                      aria-label={`Delete ${s.name}`}
                    >
                      <Trash2 />
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {s.clips.map((c) => (
                      <div key={c.id} className="flex items-center gap-1">
                        <ClipPlayer
                          path={`v1/audio/speakers/${encodeURIComponent(s.id)}/clips/${c.id}`}
                          label={`${s.name}: ${c.seconds} s`}
                        />
                        {s.clips.length > 1 && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => void removeClip(s, c.id)}
                            aria-label="Remove this recording"
                          >
                            <Trash2 />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </Page>
  )
}
