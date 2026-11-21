import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Check,
  CircleCheck,
  CircleX,
  Copy,
  ExternalLink,
  Loader2,
  Pencil,
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  Play,
  Plus,
  ShieldAlert,
  Trash2,
} from 'lucide-react'
import { useId, useState } from 'react'
import { toast } from 'sonner'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Textarea } from '@/components/ui/textarea'
import { describeError, useCall, useCalls, useInfo, useTelephony } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import { pipelinesOffered } from '@/lib/capabilities'
import type {
  CallRecord,
  CallSummary,
  PhoneAgent,
  SimulationReport,
  TelephonyOverview,
  TelephonyProvider,
} from '@/lib/api/types'
import { fmtClock, fmtMs, fmtSeconds } from '@/lib/format'
import {
  AGENT_NAME_RE,
  agentForm,
  agentUpdate,
  BAD_OUTCOMES,
  DOCS_TELEPHONY,
  LATENCY_PARTS,
  latencyShares,
  outcomeLabel,
  providerState,
  type AgentForm,
} from '@/lib/telephony'
import { cn } from '@/lib/utils'

const SERVER_DEFAULT = '__default__'
const PART_COLORS = ['bg-chart-1', 'bg-chart-2', 'bg-chart-3', 'bg-chart-4', 'bg-chart-5']

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Could not copy. Select the address instead.')
    }
  }
  return (
    <Button size="sm" variant="outline" onClick={() => void copy()} aria-label={`Copy ${label}`}>
      {copied ? <Check /> : <Copy />} {copied ? 'Copied' : 'Copy'}
    </Button>
  )
}

function ProviderCard({ p }: { p: TelephonyProvider }) {
  const state = providerState(p)
  return (
    <li className="space-y-2 px-4 py-3" data-testid={`phone-provider-${p.name}`}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-sm font-medium">{p.label}</span>
        <span
          className={cn(
            'inline-flex items-center gap-1 text-xs',
            state.ready ? 'text-good-text' : 'text-muted-foreground',
          )}
          data-testid={`phone-provider-state-${p.name}`}
        >
          {state.ready ? (
            <CircleCheck className="size-3.5" aria-hidden />
          ) : (
            <CircleX className="size-3.5" aria-hidden />
          )}
          {state.label}
        </span>
        <span className="text-xs text-muted-foreground">
          {p.protection === 'signature' ? 'Requests are checked by signature' : 'Requests are checked by credentials'}
          {p.can_place_calls ? ' · can place calls' : ''}
        </span>
        <a
          href={p.docs_url}
          target="_blank"
          rel="noreferrer"
          className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground underline underline-offset-2"
        >
          {p.label} documentation <ExternalLink className="size-3" aria-hidden />
        </a>
      </div>
      {p.problems.length > 0 && (
        <p className="text-xs leading-5 text-warning-text">
          Its endpoints stay closed until the server&apos;s configuration has: {p.problems.join('; ')}.
        </p>
      )}
      <ul className="space-y-2">
        {p.webhooks.map((hook) => (
          <li key={hook.name} className="rounded-lg border border-border bg-muted/30 p-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium">{hook.name}</span>
              <span className="text-[11px] text-muted-foreground">{hook.method}</span>
              <span className="ml-auto">
                <CopyButton text={hook.url} label={`${p.label} ${hook.name}`} />
              </span>
            </div>
            <code className="mt-1 block num text-xs break-all select-all">{hook.url}</code>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">Paste it at: {hook.where}</p>
          </li>
        ))}
      </ul>
      {p.notes.map((note) => (
        <p key={note} className="text-xs leading-5 text-muted-foreground">
          {note}
        </p>
      ))}
    </li>
  )
}

function Field({ label, children, hint }: { label: string; children: (id: string) => React.ReactNode; hint?: string }) {
  const id = useId()
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{label}</Label>
      {children(id)}
      {hint && <p className="text-xs leading-4 text-muted-foreground">{hint}</p>}
    </div>
  )
}

function AgentDialog({
  existing,
  taken,
  onClose,
}: {
  existing: PhoneAgent | null
  taken: string[]
  onClose: () => void
}) {
  const qc = useQueryClient()
  const [form, setForm] = useState<AgentForm>(() => agentForm(existing))
  const [saving, setSaving] = useState(false)
  const set = <K extends keyof AgentForm>(key: K, value: AgentForm[K]) => setForm((f) => ({ ...f, [key]: value }))
  const nameProblem = existing
    ? null
    : form.name && !AGENT_NAME_RE.test(form.name)
      ? 'Lower-case letters, digits, - and _ only.'
      : taken.includes(form.name)
        ? 'An agent with this name exists.'
        : null
  const { body, problem } = agentUpdate(form)
  const textInput = (key: keyof AgentForm, placeholder?: string) => (id: string) => (
    <Input id={id} value={form[key]} placeholder={placeholder} onChange={(e) => set(key, e.target.value)} />
  )

  const save = async () => {
    if (problem || nameProblem || !form.name) return
    setSaving(true)
    try {
      await api.putAgent(form.name, body)
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['telephony'] }),
        qc.invalidateQueries({ queryKey: ['info'] }),
      ])
      toast.success(`${form.name} saved`)
      onClose()
    } catch (err) {
      toast.error(`Could not save ${form.name}`, { description: describeError(err) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{existing ? `Edit ${existing.name}` : 'Add an agent'}</DialogTitle>
          <DialogDescription>
            An agent is what answers a call: its instructions, its voice and how it behaves on the line. Empty fields
            use the server&apos;s settings.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" hint={nameProblem ?? undefined}>
            {(id) => (
              <Input
                id={id}
                value={form.name}
                disabled={existing != null}
                aria-invalid={nameProblem != null}
                onChange={(e) => set('name', e.target.value)}
              />
            )}
          </Field>
          <Field label="Greeting" hint="Said when the call is answered. Empty: the caller speaks first.">
            {textInput('greeting', 'Hello, how can I help?')}
          </Field>
          <div className="sm:col-span-2">
            <Field label="Instructions">
              {(id) => (
                <Textarea
                  id={id}
                  rows={4}
                  value={form.instructions}
                  placeholder="You answer the phone for ..."
                  onChange={(e) => set('instructions', e.target.value)}
                />
              )}
            </Field>
          </div>
          <Field label="Voice">{textInput('voice')}</Field>
          <Field label="Language" hint="A hint for speech recognition, for example en or hi.">
            {textInput('language')}
          </Field>
          <Field label="Speech-to-text model">{textInput('sttModel')}</Field>
          <Field label="Text-to-speech model">{textInput('ttsModel')}</Field>
          <Field label="Language model">{textInput('llmModel')}</Field>
          <Field label="Pipeline">
            {(id) => (
              <Select
                value={form.pipeline || SERVER_DEFAULT}
                onValueChange={(v) => set('pipeline', v === SERVER_DEFAULT ? '' : v)}
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={SERVER_DEFAULT}>The server&apos;s default</SelectItem>
                  <SelectItem value="cascade">Cascade (STT, language model, TTS)</SelectItem>
                  <SelectItem value="audio-llm">Audio language model</SelectItem>
                  <SelectItem value="speech-to-speech">Speech to speech</SelectItem>
                </SelectContent>
              </Select>
            )}
          </Field>
          <LivePipelineField value={form.pipelineId} onChange={(v) => set('pipelineId', v)} />
          <Field label="Longest call (seconds)" hint="0: no limit.">
            {textInput('maxDurationS')}
          </Field>
          <Field label="Said at the limit">{textInput('maxDurationMessage')}</Field>
          <Field label="Silence before a prompt (seconds)" hint="0: never prompt.">
            {textInput('silenceTimeoutS')}
          </Field>
          <Field label="Prompts before hanging up">{textInput('silencePrompts')}</Field>
          <Field label="Silence prompt">{textInput('silencePrompt')}</Field>
          <Field label="Goodbye">{textInput('goodbye')}</Field>
          <Field label="Transfer number" hint="Where the agent can send the caller to reach a person.">
            {textInput('transferNumber', '+14155550100')}
          </Field>
          <Field label="Said before a transfer">{textInput('transferMessage')}</Field>
          <div className="sm:col-span-2">
            <Field label="Recording-consent announcement" hint="Said first on every call and never interrupted.">
              {textInput('recordingConsent', 'This call is transcribed for quality purposes.')}
            </Field>
          </div>
          <Field label="Outbound call reaches voicemail">
            {(id) => (
              <Select value={form.voicemail} onValueChange={(v) => set('voicemail', v)}>
                <SelectTrigger id={id} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hangup">Hang up</SelectItem>
                  <SelectItem value="leave_message">Leave a message</SelectItem>
                  <SelectItem value="ignore">Do not detect voicemail</SelectItem>
                </SelectContent>
              </Select>
            )}
          </Field>
          <Field label="Voicemail message">{textInput('voicemailMessage')}</Field>
          <div className="sm:col-span-2">
            <Field
              label="Tools (JSON)"
              hint='A list such as [{"name": "lookup_order", "description": "...", "parameters": {...}, "url": "https://..."}]. A tool with a url is called with an HTTP POST. Hanging up and transfer are built in.'
            >
              {(id) => (
                <Textarea
                  id={id}
                  rows={3}
                  className="num text-xs"
                  value={form.tools}
                  onChange={(e) => set('tools', e.target.value)}
                />
              )}
            </Field>
          </div>
        </div>
        {problem && (
          <p className="text-xs text-critical-text" role="alert">
            {problem}
          </p>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={() => void save()}
            disabled={saving || Boolean(problem) || Boolean(nameProblem) || !form.name}
          >
            {saving && <Loader2 className="animate-spin" />} {existing ? 'Save' : 'Add agent'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Numbers({ data }: { data: TelephonyOverview }) {
  const qc = useQueryClient()
  const id = useId()
  const [number, setNumber] = useState('')
  const [agent, setAgent] = useState('')
  const chosen = agent || data.default_agent || data.agents[0]?.name || ''
  const refresh = () => qc.invalidateQueries({ queryKey: ['telephony'] })
  const add = async () => {
    try {
      await api.putNumber(number.trim(), chosen)
      setNumber('')
      await refresh()
    } catch (err) {
      toast.error('Could not map the number', { description: describeError(err) })
    }
  }
  const remove = async (n: string) => {
    try {
      await api.deleteNumber(n)
      await refresh()
    } catch (err) {
      toast.error('Could not remove the number', { description: describeError(err) })
    }
  }
  return (
    <Panel
      title="Numbers"
      description={
        data.default_agent
          ? `Other numbers are answered by ${data.default_agent}`
          : 'Calls to other numbers are refused'
      }
      bodyClassName="p-0"
    >
      {data.numbers.length === 0 ? (
        <Empty icon={Phone} title="No number is mapped">
          {data.default_agent
            ? `Every call is answered by ${data.default_agent}, the default agent.`
            : 'Map the numbers you own to agents, or set telephony.default_agent in the configuration.'}
        </Empty>
      ) : (
        <ul className="divide-y divide-border" data-testid="phone-numbers">
          {data.numbers.map((n) => (
            <li key={n.number} className="flex items-center gap-3 px-4 py-2 text-sm">
              <span className="num">{n.number}</span>
              <span className="text-muted-foreground">answered by</span>
              <span className="font-medium">{n.agent}</span>
              <span className="ml-auto text-xs text-muted-foreground">
                {n.editable ? '' : 'from the configuration'}
              </span>
              {n.editable && data.editable && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => void remove(n.number)}
                  aria-label={`Remove ${n.number}`}
                >
                  <Trash2 />
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
      {data.editable && data.agents.length > 0 && (
        <div className="flex flex-wrap items-end gap-2 border-t border-border px-4 py-3">
          <div className="space-y-1.5">
            <Label htmlFor={`${id}-number`}>Number</Label>
            <Input
              id={`${id}-number`}
              className="w-48"
              placeholder="+14155550100"
              value={number}
              onChange={(e) => setNumber(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor={`${id}-agent`}>Agent</Label>
            <Select value={chosen} onValueChange={setAgent}>
              <SelectTrigger id={`${id}-agent`} className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {data.agents.map((a) => (
                  <SelectItem key={a.name} value={a.name}>
                    {a.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button size="sm" onClick={() => void add()} disabled={!number.trim() || !chosen}>
            <Plus /> Map number
          </Button>
        </div>
      )}
    </Panel>
  )
}

function CallDetail({ record }: { record: CallRecord }) {
  return (
    <div className="space-y-5 text-sm" data-testid="call-detail">
      <p className="text-xs text-muted-foreground">
        {record.direction} · {record.provider} · agent {record.agent ?? 'none'} ·{' '}
        {record.duration_s != null ? fmtSeconds(record.duration_s) : 'not connected'} ·{' '}
        <span className={cn(record.outcome && BAD_OUTCOMES.has(record.outcome) && 'text-critical-text')}>
          {outcomeLabel(record.outcome, record.status)}
        </span>
        {record.error ? ` · ${record.error}` : ''}
      </p>
      {record.summary && (
        <section>
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">Summary</h3>
          <p className="leading-6">{record.summary}</p>
        </section>
      )}
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Latency by turn: end of the caller&apos;s speech to the first audio sent
        </h3>
        {record.turns.filter((t) => t.latency.total_ms != null).length === 0 ? (
          <p className="text-xs text-muted-foreground">No turn was answered on this call.</p>
        ) : (
          <ul className="space-y-3" data-testid="call-turns">
            {record.turns
              .filter((t) => t.latency.total_ms != null)
              .map((turn) => (
                <li key={turn.index}>
                  <div className="mb-1 flex flex-wrap items-baseline gap-x-2 text-xs">
                    <span className="font-medium">Turn {turn.index + 1}</span>
                    <span className="num">{fmtMs(turn.latency.response_ms)}</span>
                    {turn.interrupted && <span className="text-warning-text">interrupted by the caller</span>}
                    {turn.kind !== 'speech' && <span className="text-muted-foreground">after {turn.kind}</span>}
                  </div>
                  <div
                    className="flex h-3 overflow-hidden rounded-sm bg-muted"
                    role="img"
                    aria-label={latencyShares(turn)
                      .map((s) => `${s.label} ${Math.round(s.ms)} ms`)
                      .join(', ')}
                  >
                    {latencyShares(turn).map((s, i) => (
                      <div
                        key={s.key}
                        className={PART_COLORS[LATENCY_PARTS.findIndex((p) => p.key === s.key)] ?? PART_COLORS[i]}
                        style={{ width: `${s.share * 100}%` }}
                        title={`${s.label}: ${Math.round(s.ms)} ms`}
                      />
                    ))}
                  </div>
                  <p className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-muted-foreground">
                    {latencyShares(turn).map((s) => (
                      <span key={s.key}>
                        <span
                          className={cn(
                            'mr-1 inline-block size-2 rounded-[2px]',
                            PART_COLORS[LATENCY_PARTS.findIndex((p) => p.key === s.key)],
                          )}
                          aria-hidden
                        />
                        {s.label} <span className="num">{fmtMs(s.ms)}</span>
                      </span>
                    ))}
                  </p>
                </li>
              ))}
          </ul>
        )}
      </section>
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">Transcript</h3>
        {record.transcript.length === 0 ? (
          <p className="text-xs text-muted-foreground">Nothing was said.</p>
        ) : (
          <ol className="space-y-1.5" data-testid="call-transcript">
            {record.transcript.map((line, i) => (
              <li key={i} className="grid grid-cols-[3.5rem_3.5rem_1fr] gap-2 leading-5">
                <span className="num text-xs text-muted-foreground">{fmtSeconds(line.t_ms / 1000)}</span>
                <span className={cn('text-xs font-medium', line.speaker === 'agent' && 'text-muted-foreground')}>
                  {line.speaker === 'agent' ? 'Agent' : 'Caller'}
                </span>
                <span>
                  {line.text}
                  {line.interrupted && <span className="ml-1 text-xs text-warning-text">(cut off by the caller)</span>}
                  {line.kind && line.kind !== 'dtmf' && (
                    <span className="ml-1 text-xs text-muted-foreground">({line.kind.replaceAll('_', ' ')})</span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">Timeline</h3>
        <ol className="space-y-0.5 text-xs" data-testid="call-timeline">
          {record.timeline.map((ev, i) => {
            const { t_ms, type, ...rest } = ev
            const detail = Object.entries(rest)
              .map(([k, v]) => `${k} ${typeof v === 'string' ? v : JSON.stringify(v)}`)
              .join(' · ')
            return (
              <li key={i} className="grid grid-cols-[3.5rem_10rem_1fr] gap-2">
                <span className="num text-muted-foreground">{fmtSeconds(t_ms / 1000)}</span>
                <span className="font-medium">{type.replaceAll('_', ' ')}</span>
                <span className="break-words text-muted-foreground">{detail}</span>
              </li>
            )
          })}
        </ol>
      </section>
    </div>
  )
}

function CallDialog({ id, onClose }: { id: string; onClose: () => void }) {
  const q = useCall(id)
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            Call <span className="num">{q.data ? `${q.data.from ?? 'unknown'} → ${q.data.to ?? 'unknown'}` : id}</span>
          </DialogTitle>
          <DialogDescription>
            {q.data ? `Started ${fmtClock(q.data.started_at ?? q.data.created_at)}` : 'Loading the call record'}
          </DialogDescription>
        </DialogHeader>
        {q.isError ? (
          <ErrorState title="Could not load the call" error={describeError(q.error)} onRetry={() => void q.refetch()} />
        ) : q.data ? (
          <CallDetail record={q.data} />
        ) : (
          <Skeleton className="h-40" />
        )}
      </DialogContent>
    </Dialog>
  )
}

function Calls({ onOpen }: { onOpen: (id: string) => void }) {
  const q = useCalls()
  const calls: CallSummary[] = q.data?.calls ?? []
  return (
    <Panel title="Recent calls" description="Kept in memory; the newest first" bodyClassName="p-0">
      {q.isLoading ? (
        <Skeleton className="m-4 h-24" />
      ) : q.isError ? (
        <ErrorState title="Could not load the calls" error={describeError(q.error)} onRetry={() => void q.refetch()} />
      ) : calls.length === 0 ? (
        <Empty icon={Phone} title="No calls yet">
          Calls appear here as they come in. The simulator below makes one without a telephony account.
        </Empty>
      ) : (
        <Table data-testid="phone-calls">
          <TableHeader>
            <TableRow>
              <TableHead>Started</TableHead>
              <TableHead>Call</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead className="text-right">Length</TableHead>
              <TableHead className="text-right">Turns</TableHead>
              <TableHead className="text-right">Response (median)</TableHead>
              <TableHead>Outcome</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {calls.map((c) => (
              <TableRow key={c.id} data-testid={`phone-call-${c.id}`}>
                <TableCell className="num whitespace-nowrap">{fmtClock(c.started_at ?? c.created_at)}</TableCell>
                <TableCell>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 text-left underline-offset-2 hover:underline"
                    onClick={() => onOpen(c.id)}
                  >
                    {c.direction === 'outbound' ? (
                      <PhoneOutgoing className="size-3.5 text-muted-foreground" aria-label="Outbound" />
                    ) : (
                      <PhoneIncoming className="size-3.5 text-muted-foreground" aria-label="Inbound" />
                    )}
                    <span className="num">
                      {c.from ?? 'unknown'} → {c.to ?? 'unknown'}
                    </span>
                  </button>
                </TableCell>
                <TableCell>{c.agent ?? DASH}</TableCell>
                <TableCell>{c.provider}</TableCell>
                <TableCell className="text-right num">
                  {c.duration_s != null ? fmtSeconds(c.duration_s) : DASH}
                </TableCell>
                <TableCell className="text-right num">{c.turns}</TableCell>
                <TableCell className="text-right num">
                  {c.latency.response_ms ? fmtMs(c.latency.response_ms.p50) : DASH}
                </TableCell>
                <TableCell className={cn(c.outcome && BAD_OUTCOMES.has(c.outcome) && 'text-critical-text')}>
                  {outcomeLabel(c.outcome, c.status)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}

const DASH = '–'

function Simulator({ data, onOpen }: { data: TelephonyOverview; onOpen: (id: string) => void }) {
  const qc = useQueryClient()
  const id = useId()
  const [agent, setAgent] = useState('')
  const [provider, setProvider] = useState('twilio')
  const [scenario, setScenario] = useState('basic')
  const [running, setRunning] = useState(false)
  const [report, setReport] = useState<SimulationReport | null>(null)
  const chosen = agent || data.default_agent || data.agents[0]?.name || ''
  const run = async () => {
    setRunning(true)
    setReport(null)
    try {
      setReport(await api.simulate({ agent: chosen || undefined, provider, scenario }))
      await qc.invalidateQueries({ queryKey: ['calls'] })
    } catch (err) {
      toast.error('The simulation did not run', { description: describeError(err) })
    } finally {
      setRunning(false)
    }
  }
  return (
    <Panel
      title="Simulate a call"
      description="A scripted caller rings the agent through a fake provider inside the server: no account, no phone"
    >
      <div className="flex flex-wrap items-end gap-2">
        <div className="space-y-1.5">
          <Label htmlFor={`${id}-agent`}>Agent</Label>
          <Select value={chosen || SERVER_DEFAULT} onValueChange={(v) => setAgent(v === SERVER_DEFAULT ? '' : v)}>
            <SelectTrigger id={`${id}-agent`} className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {data.agents.length === 0 && <SelectItem value={SERVER_DEFAULT}>A built-in test agent</SelectItem>}
              {data.agents.map((a) => (
                <SelectItem key={a.name} value={a.name}>
                  {a.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor={`${id}-provider`}>Speaks the protocol of</Label>
          <Select value={provider} onValueChange={setProvider}>
            <SelectTrigger id={`${id}-provider`} className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {data.providers.map((p) => (
                <SelectItem key={p.name} value={p.name}>
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor={`${id}-scenario`}>Scenario</Label>
          <Select value={scenario} onValueChange={setScenario}>
            <SelectTrigger id={`${id}-scenario`} className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {['basic', 'barge-in', 'silence', 'dtmf', 'noisy'].map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button size="sm" onClick={() => void run()} disabled={running}>
          {running ? <Loader2 className="animate-spin" /> : <Play />} {running ? 'Calling…' : 'Simulate a call'}
        </Button>
      </div>
      {running && (
        <p className="mt-3 text-xs text-muted-foreground" role="status">
          The call runs in real time: a scenario takes ten to twenty seconds.
        </p>
      )}
      {!data.llm_configured && (
        <p className="mt-3 text-xs text-warning-text">
          No language model is configured (llm_base_url): the agent can greet the caller but not answer.
        </p>
      )}
      {report && (
        <div className="mt-4 space-y-2 text-sm" data-testid="simulation-report" role="status">
          <p
            className={cn(
              'flex items-center gap-1.5 font-medium',
              report.passed ? 'text-good-text' : 'text-critical-text',
            )}
          >
            {report.passed ? (
              <CircleCheck className="size-4" aria-hidden />
            ) : (
              <CircleX className="size-4" aria-hidden />
            )}
            {report.passed ? 'The call went as expected' : 'The call did not go as expected'}
            <span className="num font-normal text-muted-foreground">· {fmtSeconds(report.duration_s)}</span>
          </p>
          {report.caller.response_ms && (
            <p className="text-xs text-muted-foreground">
              The caller heard the agent <span className="num">{fmtMs(report.caller.response_ms.p50)}</span> after
              finishing speaking (median; measured at the fake provider).
            </p>
          )}
          <ul className="space-y-0.5 text-xs">
            {report.checks.map((check, i) => (
              <li key={i} className={cn(!check.ok && 'text-critical-text')}>
                {check.ok ? 'Passed' : 'Failed'}: {check.type.replaceAll('_', ' ')} ({check.detail})
              </li>
            ))}
          </ul>
          {report.call_id && (
            <Button size="sm" variant="outline" onClick={() => onOpen(report.call_id as string)}>
              Open the call record
            </Button>
          )}
        </div>
      )}
    </Panel>
  )
}

/** Phone calls: what to paste into each provider's dashboard, agents, numbers, recent calls and a simulator. */
export default function PhonePage() {
  const q = useTelephony()
  const qc = useQueryClient()
  const [editing, setEditing] = useState<PhoneAgent | 'new' | null>(null)
  const [toDelete, setToDelete] = useState<PhoneAgent | null>(null)
  const [openCall, setOpenCall] = useState<string | null>(null)
  const data = q.data

  const remove = async (agent: PhoneAgent) => {
    try {
      await api.deleteAgent(agent.name)
      await qc.invalidateQueries({ queryKey: ['telephony'] })
      toast.success(`${agent.name} removed`)
    } catch (err) {
      toast.error(`Could not remove ${agent.name}`, { description: describeError(err) })
    }
  }

  if (isDemo) {
    return (
      <Page title="Phone">
        <NeedsServer title="Phone calls need a bolo server">
          A server can answer and place phone calls through Twilio, Plivo, Telnyx, Exotel or a SIP trunk and connect
          them to a voice agent. This demo runs in your browser only.
        </NeedsServer>
      </Page>
    )
  }

  return (
    <Page title="Phone">
      {!q.offered ? (
        <Panel>
          <Empty icon={Phone} title="This server has no telephony">
            It is older than phone-call support. Upgrade bolo to connect calls to voice agents.
          </Empty>
        </Panel>
      ) : q.isLoading ? (
        <Skeleton className="h-40" />
      ) : q.isError || !data ? (
        <ErrorState
          title="Could not load the phone setup"
          error={describeError(q.error)}
          onRetry={() => void q.refetch()}
        />
      ) : (
        <div className="space-y-4">
          {!data.validate_signatures && (
            <Alert data-testid="phone-unvalidated">
              <ShieldAlert />
              <AlertTitle>Requests from providers are not checked</AlertTitle>
              <AlertDescription>
                telephony.validate_signatures is off: anyone who can reach this server can start a call session. Switch
                it back on once the provider&apos;s credentials are in the configuration.
              </AlertDescription>
            </Alert>
          )}
          <Panel
            title="Providers"
            description={
              data.public_base_url
                ? `Providers reach this server at ${data.public_base_url}`
                : 'telephony.public_base_url is not set: the addresses below are a guess from this page'
            }
            bodyClassName="p-0"
            actions={
              <a
                href={DOCS_TELEPHONY}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-muted-foreground underline underline-offset-2"
              >
                Setup guide <ExternalLink className="size-3" aria-hidden />
              </a>
            }
          >
            <p className="border-b border-border px-4 py-2 text-xs leading-5 text-muted-foreground">
              Credentials are set in the server&apos;s configuration (telephony.providers) and never shown here. Paste
              each address below into the provider&apos;s dashboard.
            </p>
            <ul className="divide-y divide-border" data-testid="phone-providers">
              {data.providers.map((p) => (
                <ProviderCard key={p.name} p={p} />
              ))}
            </ul>
          </Panel>

          <div className="grid gap-4 xl:grid-cols-2">
            <Panel
              title="Agents"
              description={`${data.agents.length} agent${data.agents.length === 1 ? '' : 's'}`}
              bodyClassName="p-0"
              actions={
                <Button size="sm" onClick={() => setEditing('new')} disabled={!data.editable}>
                  <Plus /> Add agent
                </Button>
              }
            >
              {!data.editable && (
                <p className="border-b border-border px-4 py-2 text-xs text-muted-foreground">
                  {data.edit_blocked_reason ?? 'Agents cannot be changed from here.'}
                </p>
              )}
              {data.agents.length === 0 ? (
                <Empty icon={Phone} title="No agent yet">
                  An agent answers calls with its own instructions, voice and behaviour. Add one, then map a number to
                  it.
                </Empty>
              ) : (
                <ul className="divide-y divide-border" data-testid="phone-agents">
                  {data.agents.map((a) => (
                    <li
                      key={a.name}
                      className="flex items-start gap-3 px-4 py-2.5"
                      data-testid={`phone-agent-${a.name}`}
                    >
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium">
                          {a.name}
                          {a.name === data.default_agent && (
                            <span className="ml-2 text-xs font-normal text-muted-foreground">default</span>
                          )}
                          {!a.editable && (
                            <span className="ml-2 text-xs font-normal text-muted-foreground">
                              from the configuration
                            </span>
                          )}
                        </p>
                        <p className="truncate text-xs leading-5 text-muted-foreground">
                          {a.greeting ? `Greets with “${a.greeting}”` : 'Waits for the caller to speak'}
                          {a.transfer_number ? ` · transfers to ${a.transfer_number}` : ''}
                          {a.recording_consent ? ' · announces transcription' : ''}
                          {a.tools.length ? ` · ${a.tools.length} tool${a.tools.length === 1 ? '' : 's'}` : ''}
                        </p>
                      </div>
                      {a.editable && data.editable && (
                        <>
                          <Button size="sm" variant="ghost" onClick={() => setEditing(a)}>
                            <Pencil /> Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setToDelete(a)}
                            aria-label={`Remove ${a.name}`}
                          >
                            <Trash2 />
                          </Button>
                        </>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Numbers data={data} />
          </div>

          <Calls onOpen={setOpenCall} />
          <Simulator data={data} onOpen={setOpenCall} />
          <p className="text-xs leading-5 text-muted-foreground" data-testid="phone-compliance">
            {data.compliance_notice}
          </p>
        </div>
      )}

      {editing && data && (
        <AgentDialog
          key={editing === 'new' ? 'new' : editing.name}
          existing={editing === 'new' ? null : editing}
          taken={data.agents.map((a) => a.name)}
          onClose={() => setEditing(null)}
        />
      )}
      {openCall && <CallDialog id={openCall} onClose={() => setOpenCall(null)} />}
      <Dialog open={toDelete != null} onOpenChange={(o) => !o && setToDelete(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Remove <span className="num">{toDelete?.name}</span>?
            </DialogTitle>
            <DialogDescription>
              Numbers mapped to this agent fall back to the default agent, or are refused when there is none.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (toDelete) void remove(toDelete)
                setToDelete(null)
              }}
            >
              <Trash2 /> Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Page>
  )
}

/** A saved live pipeline as the agent's behaviour: it runs on each turn of the caller (docs/pipelines.md). */
function LivePipelineField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const info = useInfo().data
  const q = useQuery({
    queryKey: ['pipelines'],
    queryFn: ({ signal }) => api.pipelines(signal),
    enabled: pipelinesOffered(info),
  })
  if (!pipelinesOffered(info)) return null
  const live = (q.data?.data ?? []).filter((p) => p.mode === 'live')
  return (
    <Field label="Behaviour" hint="A live pipeline from the Pipelines page runs on each turn of the caller.">
      {(id) => (
        <Select value={value || NO_LIVE} onValueChange={(v) => onChange(v === NO_LIVE ? '' : v)}>
          <SelectTrigger id={id} className="w-full" data-testid="agent-behaviour">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NO_LIVE}>The language model answers each turn</SelectItem>
            {live.map((p) => (
              <SelectItem key={p.id} value={p.id ?? ''}>
                Live pipeline: {p.name || p.id}
              </SelectItem>
            ))}
            {value && !live.some((p) => p.id === value) && <SelectItem value={value}>{value}</SelectItem>}
          </SelectContent>
        </Select>
      )}
    </Field>
  )
}

const NO_LIVE = '__none'
