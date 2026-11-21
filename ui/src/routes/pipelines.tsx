import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  Download,
  History,
  Loader2,
  Play,
  Plus,
  Save,
  Trash2,
  Upload,
  Workflow,
  XCircle,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Badge } from '@/components/ui/badge'
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
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useCatalog, useInfo } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { CatalogModel } from '@/lib/api/types'
import { autoPull } from '@/lib/capabilities'
import {
  catalogTask,
  cleanPipeline,
  isAudio,
  isJsonOption,
  moveStep,
  optionText,
  parseImport,
  parseOption,
  pipelineProblems,
  preview,
  problemsOf,
  referencesFor,
  stepIds,
  typesFor,
  type Pipeline,
  type PipelineSimulation,
  type PipelineRun,
  type PipelineStep,
  type Problem,
  type StepSlot,
  type StepType,
  type Validation,
} from '@/lib/pipelines'
import { cn } from '@/lib/utils'

const NEW = '__new'
const DEFAULT_INPUT = '__default'
const CUSTOM_INPUT = '__custom'
const EMPTY: Pipeline = { id: '', name: '', steps: [] }
const GROUPS: [StepType['group'], string][] = [
  ['model', 'Models'],
  ['text', 'Text and the language model'],
  ['control', 'Control'],
  ['action', 'Actions'],
]

/** Pipelines: build a chain of steps, check it as you go, test it on a clip and keep it (docs/pipelines.md). */
export default function PipelinesPage() {
  const qc = useQueryClient()
  const info = useInfo().data
  const catalog = useCatalog().data?.models ?? []
  const typesQ = useQuery({ queryKey: ['pipeline-step-types'], queryFn: ({ signal }) => api.pipelineStepTypes(signal) })
  const templatesQ = useQuery({
    queryKey: ['pipeline-templates'],
    queryFn: ({ signal }) => api.pipelineTemplates(signal),
  })
  const listQ = useQuery({ queryKey: ['pipelines'], queryFn: ({ signal }) => api.pipelines(signal) })
  const types = useMemo(() => Object.fromEntries((typesQ.data?.data ?? []).map((t) => [t.type, t])), [typesQ.data])
  const [draft, setDraft] = useState<Pipeline>(EMPTY)
  const [savedId, setSavedId] = useState<string | null>(null)
  const [checked, setValidation] = useState<Validation | null>(null)
  const [run, setRun] = useState<PipelineRun | null>(null)
  const importInput = useRef<HTMLInputElement>(null)

  // check the draft as it changes (the server validates; nothing is saved)
  useEffect(() => {
    if (!draft.steps.length) return
    const ctl = new AbortController()
    const t = setTimeout(() => {
      api
        .validatePipeline(cleanPipeline(draft), ctl.signal)
        .then(setValidation)
        .catch(() => undefined)
    }, 350)
    return () => {
      clearTimeout(t)
      ctl.abort()
    }
  }, [draft])

  const open = (p: Pipeline, saved: string | null) => {
    setDraft({ ...p, steps: p.steps.map((s) => ({ ...s })) })
    setSavedId(saved)
    setRun(null)
  }

  const choose = async (id: string) => {
    if (id === NEW) return open(EMPTY, null)
    try {
      open(await api.pipeline(id), id)
    } catch (err) {
      toast.error('The pipeline could not be opened', { description: describeError(err) })
    }
  }

  const save = async () => {
    const body = cleanPipeline(draft)
    try {
      const doc = savedId ? await api.updatePipeline(savedId, body) : await api.createPipeline(body)
      toast.success(`Saved ${doc.id} (version ${doc.version})`)
      setSavedId(doc.id ?? null)
      setDraft((d) => ({ ...d, id: doc.id }))
      await qc.invalidateQueries({ queryKey: ['pipelines'] })
    } catch (err) {
      toast.error('The pipeline was not saved', { description: describeError(err) })
    }
  }

  const remove = async () => {
    if (!savedId || !window.confirm(`Delete the pipeline ${savedId} and its versions?`)) return
    try {
      await api.deletePipeline(savedId)
      open(EMPTY, null)
      await qc.invalidateQueries({ queryKey: ['pipelines'] })
    } catch (err) {
      toast.error('The pipeline was not deleted', { description: describeError(err) })
    }
  }

  const exportJson = () => {
    const body = JSON.stringify(cleanPipeline(draft), null, 2)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([body], { type: 'application/json' }))
    a.download = `${draft.id || 'pipeline'}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const importFile = async (file: File) => {
    const text = await file.text()
    let doc = parseImport(text)
    if (!doc) {
      try {
        const checked = await api.validatePipeline(text) // YAML: the server reads it
        doc = checked.pipeline as Pipeline
      } catch (err) {
        toast.error('The file is not a pipeline', { description: describeError(err) })
        return
      }
    }
    if (!doc || !Array.isArray(doc.steps)) {
      toast.error('The file is not a pipeline', { description: 'A pipeline is an object with a list of steps.' })
      return
    }
    open(doc, null)
    toast.success(`Imported ${doc.id || doc.name || file.name}: save it to keep it`)
  }

  const validation = draft.steps.length ? checked : null
  const setSteps = (steps: PipelineStep[]) => setDraft((d) => ({ ...d, steps }))
  const ids = stepIds(draft.steps)
  const errors = validation?.errors ?? []
  const warnings = validation?.warnings ?? []

  const toolbar = (
    <div className="flex flex-wrap items-center gap-2">
      <Select value={savedId ?? NEW} onValueChange={(v) => void choose(v)}>
        <SelectTrigger className="h-8 w-52" aria-label="Pipeline" data-testid="pipeline-select">
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
          <SelectItem value={NEW}>New pipeline</SelectItem>
          {(listQ.data?.data ?? []).map((p) => (
            <SelectItem key={p.id} value={p.id ?? ''}>
              {p.name || p.id}
              <span className="ml-2 text-xs text-muted-foreground">v{p.version}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value=""
        onValueChange={(v) => {
          const t = templatesQ.data?.data.find((x) => x.id === v)
          if (t) open(t, null)
        }}
      >
        <SelectTrigger className="h-8 w-44" aria-label="Start from a template" data-testid="pipeline-template">
          <SelectValue placeholder="From a template" />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
          {(templatesQ.data?.data ?? []).map((t) => (
            <SelectItem key={t.id} value={t.id ?? ''}>
              {t.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button variant="outline" size="sm" onClick={() => importInput.current?.click()}>
        <Upload /> Import
      </Button>
      <input
        ref={importInput}
        type="file"
        accept=".json,.yaml,.yml,application/json"
        className="sr-only"
        tabIndex={-1}
        aria-label="Pipeline file to import"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void importFile(f)
          e.target.value = ''
        }}
      />
      <Button variant="outline" size="sm" onClick={exportJson} disabled={!draft.steps.length}>
        <Download /> Export
      </Button>
      {savedId && (
        <Button variant="ghost" size="sm" onClick={() => void remove()} aria-label="Delete pipeline">
          <Trash2 />
        </Button>
      )}
      <Button size="sm" onClick={() => void save()} disabled={!draft.steps.length || errors.length > 0}>
        <Save /> Save
      </Button>
    </div>
  )

  return (
    <Page title="Pipelines" toolbar={toolbar}>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="grid content-start gap-4">
          <Panel
            title="Pipeline"
            description="A name, then its steps in order. Each step can use the outputs of the steps before it."
          >
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="grid gap-1 text-[13px]">
                <span className="text-muted-foreground">Id</span>
                <Input
                  value={draft.id ?? ''}
                  disabled={Boolean(savedId)}
                  placeholder="meeting-notes"
                  onChange={(e) => setDraft((d) => ({ ...d, id: e.target.value }))}
                  aria-label="Pipeline id"
                />
              </label>
              <label className="grid gap-1 text-[13px]">
                <span className="text-muted-foreground">Runs</span>
                <Select
                  value={draft.mode === 'live' ? 'live' : 'file'}
                  onValueChange={(v) => setDraft((d) => ({ ...d, mode: v }))}
                >
                  <SelectTrigger className="h-8 w-full" aria-label="Pipeline mode" data-testid="pipeline-mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="file">On a recording or a text (file)</SelectItem>
                    <SelectItem value="live">On each turn of a call or conversation (live)</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              {draft.mode !== 'live' && (
                <label className="grid gap-1 text-[13px]">
                  <span className="text-muted-foreground">
                    Triggers <span className="text-xs">(JSON: call_end, session_end, folder)</span>
                  </span>
                  <OptionText
                    value={draft.triggers}
                    type="object"
                    label="Triggers"
                    rows={1}
                    onChange={(v) =>
                      setDraft((d) => ({
                        ...d,
                        triggers: Array.isArray(v) ? (v as Record<string, unknown>[]) : undefined,
                      }))
                    }
                  />
                </label>
              )}
              <label className="grid gap-1 text-[13px]">
                <span className="text-muted-foreground">Name</span>
                <Input
                  value={draft.name ?? ''}
                  onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
                  aria-label="Pipeline name"
                />
              </label>
            </div>
            <ProblemList problems={pipelineProblems(errors)} kind="error" />
            {validation && (
              <p className="mt-3 flex items-center gap-1.5 text-xs" data-testid="pipeline-validity">
                {validation.valid ? (
                  <>
                    <CheckCircle2 className="size-3.5 text-good-text" aria-hidden /> Valid
                  </>
                ) : (
                  <>
                    <XCircle className="size-3.5 text-destructive" aria-hidden /> {errors.length} problem
                    {errors.length === 1 ? '' : 's'}
                  </>
                )}
                {warnings.length > 0 && (
                  <span className="text-muted-foreground">
                    · {warnings.length} warning{warnings.length === 1 ? '' : 's'}
                  </span>
                )}
              </p>
            )}
          </Panel>
          {draft.steps.length === 0 ? (
            <Panel>
              <Empty icon={Workflow} title="No steps yet">
                Add a first step below, or start from a template.
              </Empty>
            </Panel>
          ) : (
            draft.steps.map((step, i) => (
              <StepCard
                key={i}
                index={i}
                id={ids[i]!}
                step={step}
                steps={draft.steps}
                types={types}
                catalog={catalog}
                autoPull={autoPull(info)}
                validated={validation?.steps[i]?.id === ids[i] ? validation?.steps[i] : undefined}
                errors={problemsOf(errors, i)}
                warnings={problemsOf(warnings, i)}
                onChange={(s) => setSteps(draft.steps.map((x, j) => (j === i ? s : x)))}
                onMove={(to) => setSteps(moveStep(draft.steps, i, to))}
                onRemove={() => setSteps(draft.steps.filter((_, j) => j !== i))}
              />
            ))
          )}
          <AddStep
            types={typesFor(typesQ.data?.data ?? [], draft.mode)}
            onAdd={(type) => setSteps([...draft.steps, { type }])}
          />
        </div>
        <div className="grid content-start gap-4">
          {draft.mode === 'live' ? (
            <Simulator draft={draft} disabled={!validation?.valid} />
          ) : (
            <TestRun draft={draft} savedId={savedId} disabled={!validation?.valid} onRun={setRun} />
          )}
          {run && <RunView run={run} />}
          <RunHistory pipeline={savedId} onOpen={setRun} />
        </div>
      </div>
    </Page>
  )
}

function ProblemList({ problems, kind }: { problems: Problem[]; kind: 'error' | 'warning' }) {
  if (!problems.length) return null
  return (
    <ul className="mt-2 grid gap-1 text-xs" data-testid={`pipeline-${kind}s`}>
      {problems.map((p, i) => (
        <li
          key={`${p.path}-${i}`}
          className={cn('flex gap-1.5', kind === 'error' ? 'text-destructive' : 'text-muted-foreground')}
        >
          {kind === 'error' ? (
            <XCircle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          ) : (
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          )}
          <span>
            {p.path && <span className="num">{p.path}: </span>}
            {p.message}
          </span>
        </li>
      ))}
    </ul>
  )
}

function AddStep({ types, onAdd }: { types: StepType[]; onAdd: (type: string) => void }) {
  return (
    <Select value="" onValueChange={onAdd}>
      <SelectTrigger className="h-9 w-full border-dashed" aria-label="Add a step" data-testid="pipeline-add-step">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <Plus className="size-4" /> Add a step
        </span>
      </SelectTrigger>
      <SelectContent position="popper" align="start">
        {GROUPS.map(([group, label]) => (
          <SelectGroup key={group}>
            <SelectLabel>{label}</SelectLabel>
            {types
              .filter((t) => t.group === group)
              .map((t) => (
                <SelectItem key={t.type} value={t.type}>
                  {t.type}
                  <span className="ml-2 text-xs text-muted-foreground">{t.help}</span>
                </SelectItem>
              ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  )
}

function StepCard({
  index,
  id,
  step,
  steps,
  types,
  catalog,
  autoPull: pull,
  validated,
  errors,
  warnings,
  onChange,
  onMove,
  onRemove,
}: {
  index: number
  id: string
  step: PipelineStep
  steps: PipelineStep[]
  types: Record<string, StepType>
  catalog: CatalogModel[]
  autoPull: boolean | null
  validated?: Validation['steps'][number]
  errors: Problem[]
  warnings: Problem[]
  onChange: (s: PipelineStep) => void
  onMove: (to: number) => void
  onRemove: () => void
}) {
  const t = types[step.type]
  const task = catalogTask(t)
  const models = task ? catalog.filter((m) => m.task === task) : []
  const setOption = (name: string, value: unknown) => {
    const options = { ...(step.options ?? {}) }
    if (value === undefined) delete options[name]
    else options[name] = value
    onChange({ ...step, options })
  }
  const setInput = (name: string, value: unknown) => {
    const inputs = { ...(step.inputs ?? {}) }
    if (value === undefined) delete inputs[name]
    else inputs[name] = value
    onChange({ ...step, inputs })
  }
  return (
    <Panel
      data-testid={`pipeline-step-${index}`}
      title={
        <span className="flex items-center gap-2">
          <span className="num text-muted-foreground">{index + 1}</span>
          <Badge variant="secondary">{step.type}</Badge>
          <span className="num">{id}</span>
          {validated?.model && <span className="text-xs text-muted-foreground">{validated.model}</span>}
        </span>
      }
      description={t?.help}
      actions={
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Move up"
            onClick={() => onMove(index - 1)}
            disabled={index === 0}
          >
            <ArrowUp />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Move down"
            onClick={() => onMove(index + 1)}
            disabled={index === steps.length - 1}
          >
            <ArrowDown />
          </Button>
          <Button variant="ghost" size="icon" aria-label="Remove step" onClick={onRemove}>
            <Trash2 />
          </Button>
        </div>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="grid gap-1 text-[13px]">
          <span className="text-muted-foreground">Step id</span>
          <Input
            value={step.id ?? ''}
            placeholder={id}
            onChange={(e) => onChange({ ...step, id: e.target.value || undefined })}
            aria-label="Step id"
          />
        </label>
        {task ? (
          <label className="grid gap-1 text-[13px]">
            <span className="text-muted-foreground">Model</span>
            <CatalogSelect
              models={models}
              value={step.model && step.model !== 'auto' ? step.model : ''}
              onChange={(v) => onChange({ ...step, model: v || undefined })}
              label="Model"
              defaultLabel="auto (the server's default)"
              autoPull={pull}
              className="w-full"
            />
          </label>
        ) : t?.kind === 'llm' || t?.kind === 'decision' ? (
          <label className="grid gap-1 text-[13px]">
            <span className="text-muted-foreground">Language model</span>
            <Input
              value={step.model ?? ''}
              placeholder="the configured model (llm_model)"
              onChange={(e) => onChange({ ...step, model: e.target.value || undefined })}
              aria-label="Language model"
            />
          </label>
        ) : null}
        {(t?.inputs ?? []).map((slot) => (
          <InputField
            key={slot.name}
            slot={slot}
            value={step.inputs?.[slot.name]}
            defaults={validated?.defaults?.[slot.name]}
            references={referencesFor(steps, index, types, slot.type)}
            onChange={(v) => setInput(slot.name, v)}
          />
        ))}
        {(t?.options ?? []).map((o) => {
          const value = step.options?.[o.name]
          const label = `${o.name}${o.required ? ' *' : ''}`
          if (o.type === 'boolean')
            return (
              <label key={o.name} className="grid gap-1 text-[13px]">
                <span className="text-muted-foreground">{label}</span>
                <Select
                  value={value === undefined ? DEFAULT_INPUT : String(value)}
                  onValueChange={(v) => setOption(o.name, v === DEFAULT_INPUT ? undefined : v === 'true')}
                >
                  <SelectTrigger className="h-8 w-full" aria-label={o.name}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value={DEFAULT_INPUT}>default</SelectItem>
                    <SelectItem value="true">on</SelectItem>
                    <SelectItem value="false">off</SelectItem>
                  </SelectContent>
                </Select>
              </label>
            )
          const multiline = isJsonOption(o.type) || o.name === 'prompt' || o.name === 'template' || o.name === 'system'
          return (
            <label key={o.name} className={cn('grid gap-1 text-[13px]', multiline && 'sm:col-span-2')}>
              <span className="text-muted-foreground">
                {label}
                {o.help && <span className="ml-2 text-xs">{o.help}</span>}
                {isJsonOption(o.type) && <span className="ml-2 text-xs">(JSON)</span>}
              </span>
              {multiline ? (
                <OptionText
                  value={value}
                  type={o.type}
                  onChange={(v) => setOption(o.name, v)}
                  label={o.name}
                  rows={isJsonOption(o.type) ? 5 : 3}
                />
              ) : (
                <Input
                  value={optionText(value, o.type)}
                  onChange={(e) => setOption(o.name, parseOption(e.target.value, o.type))}
                  aria-label={o.name}
                  placeholder={o.choices.length ? o.choices.join(' | ') : o.type}
                />
              )}
            </label>
          )
        })}
        <label className="grid gap-1 text-[13px]">
          <span className="text-muted-foreground">When it fails</span>
          <Select
            value={step.on_error ?? 'fail'}
            onValueChange={(v) => onChange({ ...step, on_error: v === 'continue' ? 'continue' : undefined })}
          >
            <SelectTrigger className="h-8 w-full" aria-label="When it fails">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper">
              <SelectItem value="fail">stop the run</SelectItem>
              <SelectItem value="continue">go on without it</SelectItem>
            </SelectContent>
          </Select>
        </label>
      </div>
      <ProblemList problems={errors} kind="error" />
      <ProblemList problems={warnings} kind="warning" />
    </Panel>
  )
}

/** A JSON or long text option: edited as text, kept as typed until it parses. */
function OptionText({
  value,
  type,
  onChange,
  label,
  rows,
}: {
  value: unknown
  type: string
  onChange: (v: unknown) => void
  label: string
  rows: number
}) {
  const [text, setText] = useState(() => optionText(value, type))
  const last = useRef(value)
  useEffect(() => {
    if (value !== last.current) {
      last.current = value
      setText(optionText(value, type))
    }
  }, [value, type])
  return (
    <Textarea
      value={text}
      rows={rows}
      className="num text-xs"
      aria-label={label}
      onChange={(e) => {
        setText(e.target.value)
        const parsed = parseOption(e.target.value, type)
        last.current = parsed
        onChange(parsed)
      }}
    />
  )
}

function InputField({
  slot,
  value,
  defaults,
  references,
  onChange,
}: {
  slot: StepSlot
  value: unknown
  defaults?: string[]
  references: { ref: string; label: string; type: string }[]
  onChange: (v: unknown) => void
}) {
  const known = typeof value === 'string' && references.some((r) => r.ref === value)
  const mode = value === undefined ? DEFAULT_INPUT : known ? String(value) : CUSTOM_INPUT
  const defaultText = defaults?.length ? `default: ${defaults[0]!.replace(/[{}]/g, '')}` : 'none'
  return (
    <label className="grid gap-1 text-[13px]">
      <span className="text-muted-foreground">
        input {slot.name}
        <span className="ml-1 text-xs">({slot.type})</span>
        {slot.required && !defaults?.length ? ' *' : ''}
      </span>
      <Select
        value={mode}
        onValueChange={(v) => onChange(v === DEFAULT_INPUT ? undefined : v === CUSTOM_INPUT ? '' : v)}
      >
        <SelectTrigger className="h-8 w-full" aria-label={`input ${slot.name}`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper">
          <SelectItem value={DEFAULT_INPUT}>{defaultText}</SelectItem>
          {references.map((r) => (
            <SelectItem key={r.ref} value={r.ref}>
              {r.label}
              <span className="ml-2 text-xs text-muted-foreground">{r.type}</span>
            </SelectItem>
          ))}
          <SelectItem value={CUSTOM_INPUT}>a text or template…</SelectItem>
        </SelectContent>
      </Select>
      {mode === CUSTOM_INPUT && (
        <Input
          value={typeof value === 'string' ? value : JSON.stringify(value)}
          onChange={(e) => onChange(e.target.value)}
          aria-label={`input ${slot.name} text`}
          placeholder="Text with {{steps.id.output}} references"
        />
      )}
    </label>
  )
}

function TestRun({
  draft,
  savedId,
  disabled,
  onRun,
}: {
  draft: Pipeline
  savedId: string | null
  disabled: boolean
  onRun: (r: PipelineRun) => void
}) {
  const qc = useQueryClient()
  const [clip, setClip] = useState<Clip | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true)
    try {
      const r = await api.runPipeline(
        { file: clip?.blob, filename: clip?.name, text: text || undefined },
        undefined,
        cleanPipeline(draft),
      )
      onRun(r)
      if (r.status !== 'ok') toast.error('The run stopped at a step', { description: r.error?.message })
      await qc.invalidateQueries({ queryKey: ['pipeline-runs'] })
    } catch (err) {
      toast.error('The pipeline could not run', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel
      title="Test run"
      description={
        savedId
          ? 'Runs the pipeline as it is in the builder (saved or not) on a clip or a text.'
          : 'Runs the pipeline in the builder on a clip or a text; nothing is saved.'
      }
    >
      <div className="grid gap-3">
        <ClipSlot label="Clip" clip={clip} onChange={setClip} disabled={busy} testId="pipeline-clip" />
        <label className="grid gap-1 text-[13px]">
          <span className="text-muted-foreground">Text (input.text)</span>
          <Textarea value={text} rows={2} onChange={(e) => setText(e.target.value)} aria-label="Run text" />
        </label>
        <Button onClick={() => void go()} disabled={busy || disabled || (!clip && !text)} data-testid="pipeline-run">
          {busy ? <Loader2 className="animate-spin" /> : <Play />} {busy ? 'Running…' : 'Run'}
        </Button>
      </div>
    </Panel>
  )
}

const STATUS_CLASS: Record<string, string> = {
  ok: 'text-good-text',
  error: 'text-destructive',
  skipped: 'text-muted-foreground',
  pending: 'text-muted-foreground',
}

function RunView({ run }: { run: PipelineRun }) {
  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          Run <span className="num text-xs text-muted-foreground">{run.id}</span>
        </span>
      }
      description={`${run.status}${run.ms != null ? ` · ${(run.ms / 1000).toFixed(2)} s` : ''}${
        run.stopped ? ` · stopped: ${run.stopped}` : ''
      }`}
    >
      <RunSteps run={run} testId="pipeline-run-steps" />
      {run.outputs && Object.keys(run.outputs).length > 0 && (
        <div className="mt-3 border-t pt-3">
          <p className="mb-1 text-xs text-muted-foreground">Outputs</p>
          <Outputs values={run.outputs} />
        </div>
      )}
    </Panel>
  )
}

function RunSteps({ run, testId }: { run: PipelineRun; testId?: string }) {
  return (
    <ol className="grid gap-2" data-testid={testId}>
      {run.steps.map((s, i) => (
        <li
          key={`${s.id}-${s.speaker ?? ''}-${i}`}
          className="rounded-md border p-2 text-[13px]"
          style={{ marginLeft: `${s.depth * 16}px` }}
          data-testid={`run-step-${s.id}`}
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="num">{s.id}</span>
            {s.speaker && <Badge variant="outline">speaker {s.speaker}</Badge>}
            <span className="text-xs text-muted-foreground">{s.type}</span>
            {s.model && <span className="text-xs text-muted-foreground">{s.model}</span>}
            <span className={cn('ml-auto text-xs', STATUS_CLASS[s.status])}>{s.status}</span>
            {s.ms != null && <span className="num text-xs text-muted-foreground">{s.ms.toFixed(0)} ms</span>}
          </div>
          {s.error && <p className="mt-1 text-xs text-destructive">{s.error.message}</p>}
          {s.output && <Outputs values={s.output} />}
        </li>
      ))}
    </ol>
  )
}

function Outputs({ values }: { values: Record<string, unknown> }) {
  return (
    <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
      {Object.entries(values).map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="min-w-0 break-words">
            {isAudio(v) ? <AudioOut seconds={v.seconds} url={v.url} /> : <span className="num">{preview(v)}</span>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function AudioOut({ seconds, url }: { seconds: number; url?: string }) {
  const [src, setSrc] = useState<string | null>(null)
  useEffect(() => () => void (src && URL.revokeObjectURL(src)), [src])
  const load = async () => {
    if (!url) return
    try {
      setSrc(URL.createObjectURL(await api.pipelineFile(url)))
    } catch (err) {
      toast.error('The audio could not be loaded', { description: describeError(err) })
    }
  }
  if (src) return <audio src={src} controls autoPlay className="h-8 w-full max-w-xs" />
  return (
    <span className="flex items-center gap-2">
      audio, {seconds.toFixed(1)} s
      {url && (
        <Button variant="ghost" size="sm" className="h-6 px-2" onClick={() => void load()} aria-label="Play">
          <Play className="size-3" />
        </Button>
      )}
    </span>
  )
}

function RunHistory({ pipeline, onOpen }: { pipeline: string | null; onOpen: (r: PipelineRun) => void }) {
  const q = useQuery({
    queryKey: ['pipeline-runs', pipeline],
    queryFn: ({ signal }) => api.pipelineRuns(pipeline ?? undefined, signal),
  })
  const runs = q.data?.data ?? []
  return (
    <Panel title="Run history" description={pipeline ? `Runs of ${pipeline}` : 'Recent runs of every pipeline'}>
      {runs.length === 0 ? (
        <Empty icon={History} title="No runs yet" />
      ) : (
        <ul className="grid gap-1 text-[13px]" data-testid="pipeline-history">
          {runs.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-muted"
                onClick={() => onOpen(r)}
              >
                <span className={cn('text-xs', STATUS_CLASS[r.status])}>{r.status}</span>
                <span className="num">{r.pipeline ?? 'not saved'}</span>
                <span className="ml-auto text-xs text-muted-foreground">
                  {r.created_at.replace('T', ' ').replace('Z', '')}
                  {r.ms != null ? ` · ${(r.ms / 1000).toFixed(1)} s` : ''}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

/** A live pipeline played turn by turn: typed turns, or a recording of the caller cut at its pauses. */
function Simulator({ draft, disabled }: { draft: Pipeline; disabled: boolean }) {
  const [turns, setTurns] = useState('')
  const [instructions, setInstructions] = useState('')
  const [clip, setClip] = useState<Clip | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<PipelineSimulation | null>(null)
  const go = async () => {
    setBusy(true)
    try {
      const lines = turns
        .split('\n')
        .map((t) => t.trim())
        .filter(Boolean)
      setResult(
        await api.simulatePipeline(cleanPipeline(draft), {
          turns: lines,
          file: lines.length ? undefined : clip?.blob,
          filename: clip?.name,
          instructions: instructions || undefined,
        }),
      )
    } catch (err) {
      toast.error('The conversation could not be simulated', { description: describeError(err) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <Panel
        title="Simulate a conversation"
        description="Each line is one turn of the caller, run through the pipeline as in a call; or a recording of the caller, cut at its pauses."
      >
        <div className="grid gap-3">
          <label className="grid gap-1 text-[13px]">
            <span className="text-muted-foreground">The caller says (one turn per line)</span>
            <Textarea
              value={turns}
              rows={4}
              onChange={(e) => setTurns(e.target.value)}
              aria-label="Caller turns"
              placeholder={'Hi, my bill is wrong\nI want to speak to a person'}
            />
          </label>
          <ClipSlot label="Or a recording" clip={clip} onChange={setClip} disabled={busy} testId="pipeline-sim-clip" />
          <label className="grid gap-1 text-[13px]">
            <span className="text-muted-foreground">Session instructions (for respond steps)</span>
            <Input
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              aria-label="Session instructions"
            />
          </label>
          <Button
            onClick={() => void go()}
            disabled={busy || disabled || (!turns.trim() && !clip)}
            data-testid="pipeline-simulate"
          >
            {busy ? <Loader2 className="animate-spin" /> : <Play />} {busy ? 'Simulating…' : 'Simulate'}
          </Button>
        </div>
      </Panel>
      {result && (
        <Panel
          title="Conversation"
          description={
            result.ended ? `Ended: ${result.ended.action}${result.ended.to ? ` to ${result.ended.to}` : ''}` : undefined
          }
        >
          <ol className="grid gap-3" data-testid="pipeline-sim-turns">
            {result.turns.map((t, i) => (
              <li key={i} className="grid gap-1 text-[13px]">
                <p>
                  <span className="text-muted-foreground">Caller: </span>
                  {t.user}
                </p>
                <p>
                  <span className="text-muted-foreground">Agent: </span>
                  {t.agent || <span className="text-muted-foreground">(nothing)</span>}
                </p>
                {(t.run.actions ?? []).length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {(t.run.actions ?? []).map((a) => a.action + (a.to ? ` ${a.to}` : '')).join(' → ')}
                  </p>
                )}
                <RunSteps run={t.run} />
              </li>
            ))}
          </ol>
        </Panel>
      )}
    </>
  )
}
