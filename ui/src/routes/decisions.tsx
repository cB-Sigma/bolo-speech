import { ListChecks, Loader2, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useInfo, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { AudioDecision, DecisionAnswer, DecisionQuestion } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'

type Kind = DecisionQuestion['type']

/** One question as typed on the page: the answers are one per line (options of a choice, levels of a score). */
export interface DraftQuestion {
  type: Kind
  question: string
  answers: string
}

const KIND_LABEL: Record<Kind, string> = { yes_no: 'Yes or no', choice: 'One of these', score: 'A level on a scale' }

/** The page's questions as the request's `questions` (ids q1, q2, ...); an error message when one is incomplete. */
export function buildQuestions(drafts: DraftQuestion[]): Record<string, DecisionQuestion> | string {
  const out: Record<string, DecisionQuestion> = {}
  for (const [i, d] of drafts.entries()) {
    const question = d.question.trim()
    if (!question) return `Question ${i + 1} is empty`
    const lines = d.answers
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
    if (d.type === 'yes_no') out[`q${i + 1}`] = { type: 'yes_no', question }
    else if (lines.length < 2)
      return `Question ${i + 1} needs at least two ${d.type === 'choice' ? 'options' : 'levels'}`
    else if (d.type === 'choice') out[`q${i + 1}`] = { type: 'choice', question, options: lines }
    else if (lines.length > 10) return `Question ${i + 1}: a scale has at most 10 levels`
    else out[`q${i + 1}`] = { type: 'score', question, levels: lines }
  }
  return out
}

/** "yes (0.93)", "billing (0.98)", "Today (level 1.2 of 0–2)". */
export function answerText(a: DecisionAnswer): string {
  if (a.type === 'yes_no') return `${a.answer ? 'yes' : 'no'} (probability of yes ${a.probability.toFixed(2)})`
  if (a.type === 'choice') return `${a.answer} (${(a.probabilities[a.answer] ?? 0).toFixed(2)})`
  return `${a.label} (level ${a.score.toFixed(2)} of 0–${a.probabilities.length - 1})`
}

function confidence(a: DecisionAnswer): number {
  if (a.type === 'yes_no') return Math.abs(a.probability - 0.5) * 2
  return a.confidence
}

/** Typed questions about a recording or a text (`POST /v1/audio/decisions`), with any decision model. */
export default function DecisionsPage() {
  const info = useInfo()
  const models = useModels('decision').models
  const [model, setModel] = useState('')
  const [clip, setClip] = useState<Clip | null>(null)
  const [text, setText] = useState('')
  const [drafts, setDrafts] = useState<DraftQuestion[]>([
    { type: 'yes_no', question: 'Is the speaker asking for help?', answers: '' },
  ])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ body: AudioDecision; asked: DraftQuestion[] } | null>(null)
  const defaultId = info.data?.capabilities?.decision?.default_model ?? null
  const chosen = models.find((m) => m.id === (model || defaultId)) ?? null

  const update = (i: number, patch: Partial<DraftQuestion>) =>
    setDrafts((ds) => ds.map((d, j) => (j === i ? { ...d, ...patch } : d)))

  const ask = async () => {
    const questions = buildQuestions(drafts)
    if (typeof questions === 'string') {
      setError(questions)
      return
    }
    if (!clip && !text.trim()) {
      setError('Give a recording, a text, or both')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const body = await api.audioDecisions(questions, {
        file: clip?.blob,
        name: clip?.name,
        text: text.trim() || undefined,
        model: model || undefined,
      })
      setResult({ body, asked: drafts })
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Page title="Decisions">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
        <Panel
          title="Ask"
          description="A decision model answers each question with one of its answers and a probability; it writes no text"
          bodyClassName="grid gap-3"
        >
          <CatalogSelect
            models={models}
            value={model}
            onChange={setModel}
            label="Decision model"
            defaultLabel={defaultId ? `Server default (${defaultId})` : 'Server default'}
            disabled={busy}
            autoPull={autoPull(info.data)}
            testId="decision-model"
          />
          {chosen?.license_note && <p className="text-xs text-warning-text">{chosen.license_note}</p>}
          <ClipSlot
            label="Recording (at most 30 s)"
            clip={clip}
            onChange={setClip}
            disabled={busy}
            testId="decision-clip"
          />
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Text the questions are about, with the recording or instead of it (a transcript, a message)"
            aria-label="Text"
            rows={2}
            disabled={busy}
            data-testid="decision-text"
          />
          {drafts.map((d, i) => (
            <div
              key={i}
              className="grid gap-1.5 rounded-lg border border-border p-2.5"
              data-testid={`decision-q${i + 1}`}
            >
              <div className="flex items-center gap-2">
                <Input
                  value={d.question}
                  onChange={(e) => update(i, { question: e.target.value })}
                  placeholder="The question"
                  aria-label={`Question ${i + 1}`}
                  disabled={busy}
                />
                <Select value={d.type} onValueChange={(v) => update(i, { type: v as Kind })} disabled={busy}>
                  <SelectTrigger className="h-9 w-44 shrink-0" aria-label={`Answer type of question ${i + 1}`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" align="end">
                    {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
                      <SelectItem key={k} value={k}>
                        {KIND_LABEL[k]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove question ${i + 1}`}
                  disabled={busy || drafts.length === 1}
                  onClick={() => setDrafts((ds) => ds.filter((_, j) => j !== i))}
                >
                  <Trash2 />
                </Button>
              </div>
              {d.type !== 'yes_no' && (
                <Textarea
                  value={d.answers}
                  onChange={(e) => update(i, { answers: e.target.value })}
                  placeholder={d.type === 'choice' ? 'One option per line' : 'One level per line, lowest first'}
                  aria-label={`${d.type === 'choice' ? 'Options' : 'Levels'} of question ${i + 1}`}
                  rows={3}
                  disabled={busy}
                />
              )}
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={busy || drafts.length >= 32}
              onClick={() => setDrafts((ds) => [...ds, { type: 'yes_no', question: '', answers: '' }])}
            >
              <Plus /> Question
            </Button>
            <Button size="sm" onClick={() => void ask()} disabled={busy} data-testid="decision-ask">
              {busy ? <Loader2 className="animate-spin" /> : <ListChecks />} Ask
            </Button>
          </div>
        </Panel>
        <Panel
          title="Answers"
          description={
            result ? `${result.body.model}` : 'The most likely answer to each question, with its probability'
          }
          data-testid="decision-answers"
        >
          {error && (
            <Alert variant="destructive" className="mb-3" data-testid="decision-error">
              <AlertTitle>No answer</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {result ? (
            <ol className="grid gap-3">
              {Object.entries(result.body.answers).map(([id, a]) => {
                const i = Number(id.slice(1)) - 1
                return (
                  <li key={id} className="grid gap-1 text-[13px]">
                    <span className="text-muted-foreground">{result.asked[i]?.question ?? id}</span>
                    <span className="font-medium">{answerText(a)}</span>
                    <span className="h-1.5 rounded bg-muted">
                      <span
                        className="block h-1.5 rounded bg-primary"
                        style={{ width: `${Math.round(confidence(a) * 100)}%` }}
                        aria-label={`confidence ${confidence(a).toFixed(2)}`}
                      />
                    </span>
                  </li>
                )
              })}
            </ol>
          ) : (
            <Empty icon={ListChecks} title="No questions asked yet" className="py-8">
              Upload or record a clip, or write a text, then ask. The probabilities are the model&apos;s, not calibrated
              for your data.
            </Empty>
          )}
        </Panel>
      </div>
    </Page>
  )
}
