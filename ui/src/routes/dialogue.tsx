import { Download, MessagesSquare, Play, Plus, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { Empty, Panel } from '@/components/panel'
import { useVoices } from '@/components/voices/use-voices'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { describeError, useSelectedModel } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import { download } from '@/lib/bench'
import { canClone, followsInstructions } from '@/lib/capabilities'
import {
  dialogueBody,
  parseScript,
  parseSegments,
  REFERENCE_CLIPS,
  speakersOf,
  speaksDialogues,
  toScript,
  type DialogueMode,
  type DialogueSegment,
  type DialogueTurn,
} from '@/lib/dialogue'
import { fmtSeconds } from '@/lib/format'

const STARTER: DialogueTurn[] = [
  { speaker: 'Asha', text: 'Welcome back to the show. Today we are talking about speech on small computers.' },
  { speaker: 'Ravi', text: 'Thanks for having me. It runs better than most people expect.' },
  { speaker: 'Asha', text: 'Then let us hear how it sounds.' },
]

const DEFAULT = '__default__'

interface Result {
  url: string
  bytes: Uint8Array
  mode: string
  model: string
  segments: DialogueSegment[]
  instructions: string | null
}

/** The Dialogue builder: turns of speaker and text, a voice per speaker, spoken as one recording. */
export default function DialoguePage() {
  const { models, model, select, isLoading } = useSelectedModel('tts')
  const library = useVoices()
  const [turns, setTurns] = useState<DialogueTurn[]>(STARTER)
  const [voices, setVoices] = useState<Record<string, string>>({})
  const [mode, setMode] = useState<DialogueMode>('auto')
  const [pause, setPause] = useState(300)
  const [instructions, setInstructions] = useState('')
  const [editing, setEditing] = useState<'turns' | 'script'>('turns')
  const [script, setScript] = useState(toScript(STARTER))
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const abort = useRef<AbortController | null>(null)
  const speakers = useMemo(() => speakersOf(turns), [turns])
  const native = speaksDialogues(model)
  const saved = canClone(model) || native ? (library.data?.voices ?? []) : []
  const own = model?.voices ?? []

  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => () => void (result && URL.revokeObjectURL(result.url)), [result])

  const update = (i: number, patch: Partial<DialogueTurn>) =>
    setTurns((ts) => ts.map((t, k) => (k === i ? { ...t, ...patch } : t)))

  const speak = async () => {
    if (!model) return
    const current = editing === 'script' ? parseScript(script) : turns
    if (editing === 'script') setTurns(current)
    const body = dialogueBody(model.id, current, voices, {
      mode,
      pauseMs: pause,
      instructions: followsInstructions(model) ? instructions : undefined,
    })
    if (!body.turns.length) {
      toast.error('Add at least one turn with a speaker and text')
      return
    }
    abort.current?.abort()
    const ctl = new AbortController()
    abort.current = ctl
    setBusy(true)
    try {
      const res = await api.dialogue(body, ctl.signal)
      const blob = await res.blob()
      setResult({
        url: URL.createObjectURL(blob),
        bytes: new Uint8Array(await blob.arrayBuffer()),
        mode: res.headers.get('x-bolo-dialogue') ?? 'stitched',
        model: res.headers.get('x-bolo-model') ?? model.id,
        segments: parseSegments(res.headers.get('x-bolo-segments')),
        instructions: res.headers.get('x-bolo-instructions'),
      })
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        toast.error('The dialogue could not be spoken', { description: describeError(err) })
      }
    } finally {
      setBusy(false)
    }
  }

  const toolbar = (
    <>
      <ModelPicker
        models={models}
        value={model?.id ?? null}
        onChange={select}
        loading={isLoading}
        disabled={busy}
        className="w-60 max-sm:min-w-44 max-sm:flex-1"
      />
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={mode}
        onValueChange={(v) => v && setMode(v as DialogueMode)}
        aria-label="How to speak it"
        disabled={busy}
        className="ml-auto"
      >
        <ToggleGroupItem value="auto" className="px-3 text-xs" title="In one take when the model can, else stitched">
          Auto
        </ToggleGroupItem>
        <ToggleGroupItem value="native" className="px-3 text-xs" disabled={!native} title="The model speaks it all">
          One take
        </ToggleGroupItem>
        <ToggleGroupItem value="stitched" className="px-3 text-xs" title="Each turn on its own, joined">
          Stitched
        </ToggleGroupItem>
      </ToggleGroup>
    </>
  )

  return (
    <Page title="Dialogue" toolbar={toolbar}>
      {isDemo && (
        <NeedsServer title="Dialogues are spoken by the server" className="mb-4">
          This demo runs in your browser; a bolo server speaks dialogues with any of its voices.
        </NeedsServer>
      )}
      {model && !model.installed && (
        <Alert className="mb-4">
          <AlertTitle>
            <span className="num">{model.id}</span> is not installed
          </AlertTitle>
          <AlertDescription>Pull it from the Models screen first.</AlertDescription>
        </Alert>
      )}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="Turns"
            description={`${turns.length} turns, ${speakers.length} speakers`}
            actions={
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={editing}
                onValueChange={(v) => {
                  if (v === 'script') setScript(toScript(turns))
                  else if (v === 'turns') setTurns(parseScript(script))
                  if (v) setEditing(v as 'turns' | 'script')
                }}
                aria-label="Edit as"
              >
                <ToggleGroupItem value="turns" className="px-3 text-xs">
                  Turns
                </ToggleGroupItem>
                <ToggleGroupItem value="script" className="px-3 text-xs">
                  Script
                </ToggleGroupItem>
              </ToggleGroup>
            }
          >
            {editing === 'script' ? (
              <Textarea
                value={script}
                onChange={(e) => setScript(e.target.value)}
                aria-label="Dialogue script"
                className="min-h-56 resize-y font-mono text-[13px] leading-6"
                placeholder={'Asha: Hello.\nRavi: Hi there.'}
                data-testid="dialogue-script"
              />
            ) : (
              <div className="grid gap-2" data-testid="dialogue-turns">
                {turns.map((t, i) => (
                  <div
                    key={i}
                    className="grid grid-cols-[8rem_minmax(0,1fr)_auto] items-start gap-2 max-sm:grid-cols-1"
                  >
                    <Input
                      value={t.speaker}
                      onChange={(e) => update(i, { speaker: e.target.value })}
                      aria-label={`Speaker of turn ${i + 1}`}
                      list="dialogue-speakers"
                      className="h-8 text-[13px]"
                      maxLength={40}
                    />
                    <Textarea
                      value={t.text}
                      onChange={(e) => update(i, { text: e.target.value })}
                      aria-label={`Text of turn ${i + 1}`}
                      className="min-h-8 resize-y py-1.5 text-[13px] leading-5"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setTurns((ts) => ts.filter((_, k) => k !== i))}
                      aria-label={`Remove turn ${i + 1}`}
                      disabled={turns.length <= 1}
                    >
                      <Trash2 />
                    </Button>
                  </div>
                ))}
                <datalist id="dialogue-speakers">
                  {speakers.map((s) => (
                    <option key={s} value={s} />
                  ))}
                </datalist>
                <div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setTurns((ts) => [
                        ...ts,
                        { speaker: speakers[ts.length % Math.max(1, speakers.length)] ?? 'Speaker 1', text: '' },
                      ])
                    }
                  >
                    <Plus /> Add a turn
                  </Button>
                </div>
              </div>
            )}
            {followsInstructions(model) && (
              <div className="mt-3 grid gap-1.5">
                <label htmlFor="dialogue-instructions" className="text-xs text-muted-foreground">
                  Instructions for the whole dialogue
                </label>
                <Textarea
                  id="dialogue-instructions"
                  value={instructions}
                  onChange={(e) => setInstructions(e.target.value)}
                  placeholder="Relaxed and friendly, like a radio interview."
                  className="min-h-12 resize-y text-[13px]"
                  maxLength={4096}
                />
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Button
                onClick={() => void speak()}
                disabled={busy || !model?.installed || isDemo}
                className="h-9 min-w-24"
                data-testid="dialogue-speak"
              >
                <Play className="fill-current" /> {busy ? 'Speaking…' : 'Speak'}
              </Button>
              {busy && (
                <Button variant="outline" className="h-9" onClick={() => abort.current?.abort()}>
                  Stop
                </Button>
              )}
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                Pause between turns
                <Input
                  type="number"
                  min={0}
                  max={5000}
                  step={50}
                  value={pause}
                  onChange={(e) => setPause(Math.max(0, Math.min(5000, Number(e.target.value) || 0)))}
                  className="h-8 w-20 text-[13px]"
                  aria-label="Pause between turns in milliseconds"
                  disabled={mode === 'native'}
                />
                ms
              </label>
            </div>
          </Panel>
        </div>
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="Voices"
            description={
              native
                ? 'This model speaks the dialogue in one take, imitating each voice from a recording: pick a saved voice or a reference clip.'
                : 'Each speaker speaks with a voice of the model, or a saved voice when the model clones voices.'
            }
          >
            {speakers.length === 0 ? (
              <Empty icon={MessagesSquare} title="No speakers yet" />
            ) : (
              <div className="grid gap-2" data-testid="dialogue-voices">
                {speakers.map((s, k) => (
                  <div key={s} className="flex items-center gap-2">
                    <span className="w-28 truncate text-[13px]">{s}</span>
                    <Select
                      value={voices[s] || DEFAULT}
                      onValueChange={(v) => setVoices((vs) => ({ ...vs, [s]: v === DEFAULT ? '' : v }))}
                    >
                      <SelectTrigger className="h-8 flex-1" aria-label={`Voice of ${s}`}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent position="popper" align="start" className="max-h-72">
                        <SelectItem value={DEFAULT}>
                          {native ? `Reference clip ${REFERENCE_CLIPS[k % 2]}` : 'Model default'}
                        </SelectItem>
                        {native ? (
                          <SelectGroup>
                            <SelectLabel className="text-xs">Reference clips</SelectLabel>
                            {REFERENCE_CLIPS.map((c) => (
                              <SelectItem key={c} value={c}>
                                {c}
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        ) : (
                          own.length > 0 && (
                            <SelectGroup>
                              <SelectLabel className="text-xs">Voices of the model</SelectLabel>
                              {own.map((v) => (
                                <SelectItem key={v.id} value={v.id}>
                                  {v.name}
                                  <span className="ml-2 num text-xs text-muted-foreground">{v.language}</span>
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          )
                        )}
                        {saved.length > 0 && (
                          <SelectGroup>
                            <SelectLabel className="text-xs">Saved voices (cloned)</SelectLabel>
                            {saved.map((v) => (
                              <SelectItem key={v.id} value={v.id}>
                                {v.name}
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>
            )}
          </Panel>
          <Panel title="Result">
            {!result ? (
              <p className="text-[13px] leading-5 text-muted-foreground">The recording appears here.</p>
            ) : (
              <div className="grid gap-3" data-testid="dialogue-result">
                <audio controls src={result.url} className="w-full" />
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  <dt className="text-muted-foreground">Model</dt>
                  <dd className="text-right num break-all">{result.model}</dd>
                  <dt className="text-muted-foreground">Spoken</dt>
                  <dd className="text-right">{result.mode === 'native' ? 'in one take' : 'turn by turn, stitched'}</dd>
                </dl>
                {result.instructions === 'ignored' && (
                  <p className="text-xs text-warning-text">
                    This model does not follow instructions; they were ignored.
                  </p>
                )}
                {result.segments.length > 0 && (
                  <ol className="grid gap-1 text-xs text-muted-foreground">
                    {result.segments.map((g) => (
                      <li key={g.turn} className="flex justify-between gap-2">
                        <span className="truncate">{g.speaker}</span>
                        <span className="num">
                          {fmtSeconds(g.start)} – {fmtSeconds(g.end)}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
                <div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => download(`bolo-dialogue-${result.model}.wav`, result.bytes, 'audio/wav')}
                  >
                    <Download /> Save WAV
                  </Button>
                </div>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </Page>
  )
}
