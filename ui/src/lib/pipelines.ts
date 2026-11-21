/** Pipelines (docs/pipelines.md): the wire types and the pure helpers the builder uses. */

import type { Task } from '@/lib/api/types'

export interface PipelineStep {
  id?: string
  type: string
  model?: string
  inputs?: Record<string, unknown>
  options?: Record<string, unknown>
  when?: unknown
  on_error?: 'fail' | 'continue'
}

export interface Pipeline {
  id?: string
  name?: string
  description?: string
  mode?: string
  vars?: Record<string, unknown>
  steps: PipelineStep[]
  triggers?: Record<string, unknown>[]
  outputs?: Record<string, unknown>
  version?: number
  created_at?: string
  updated_at?: string
}

export interface StepSlot {
  name: string
  type: string
  required: boolean
  default: string | null
  help: string
}

export interface StepOption {
  name: string
  type: string
  help: string
  required: boolean
  choices: string[]
}

export interface StepType {
  type: string
  /** The catalog task of its model; 'llm' for the configured language model; null: no model. */
  kind: string | null
  group: 'model' | 'text' | 'control' | 'action'
  help: string
  inputs: StepSlot[]
  options: StepOption[]
  outputs: Record<string, string>
  live: boolean
  live_only: boolean
}

export interface Problem {
  path: string
  message: string
}

export interface ValidatedStep {
  id: string
  type: string
  model: string | null
  inputs: Record<string, unknown>
  defaults: Record<string, string[]>
  outputs: Record<string, string>
  path: string
}

export interface Validation {
  valid: boolean
  errors: Problem[]
  warnings: Problem[]
  steps: ValidatedStep[]
}

export interface AudioOutput {
  type: 'audio'
  seconds: number
  sample_rate: number
  file?: string
  url?: string
}

export interface RunStep {
  id: string
  type: string
  model: string | null
  status: 'running' | 'ok' | 'error' | 'skipped' | 'pending' | 'cancelled'
  ms: number | null
  output: Record<string, unknown> | null
  error: { message: string; code: string } | null
  depth: number
  speaker?: string
}

export interface PipelineRun {
  id: string
  object: 'pipeline.run'
  pipeline: string | null
  version: number | null
  status: 'queued' | 'running' | 'ok' | 'error' | 'cancelled'
  created_at: string
  started_at: string | null
  finished_at: string | null
  ms: number | null
  input: Record<string, unknown>
  steps: RunStep[]
  outputs: Record<string, unknown> | null
  error: { step: string | null; message: string; code: string } | null
  files: { name: string; url: string }[]
  stopped?: string
  job_id?: string
  trigger?: string
  actions?: { action: string; text?: string; to?: string | null }[]
  warnings?: Problem[]
}

/** Step types whose models are catalog models of this task. */
export function catalogTask(t: StepType | undefined): Task | null {
  if (!t?.kind || t.kind === 'llm' || t.kind === 'decision') return null
  return t.kind as Task
}

/** The id a step gets when it names none (as the server numbers them). */
export function stepIds(steps: PipelineStep[]): string[] {
  const used = new Set(steps.map((s) => s.id).filter((x): x is string => Boolean(x)))
  const counter: Record<string, number> = {}
  return steps.map((s) => {
    if (s.id) return s.id
    const base = s.type.replace(/-/g, '_')
    let id: string
    do {
      counter[base] = (counter[base] ?? 0) + 1
      id = counter[base] === 1 ? base : `${base}_${counter[base]}`
    } while (used.has(id))
    used.add(id)
    return id
  })
}

/** Can a value of type `value` go into a slot of type `slot`? (bolo.pipelines.definition.accepts) */
export function accepts(slot: string, value: string): boolean {
  if (slot === 'any' || value === 'any' || slot === value) return true
  if (slot === 'text') return ['transcript', 'number', 'boolean'].includes(value)
  if (slot === 'list') return value === 'turns'
  if (slot === 'turns') return value === 'list'
  if (slot === 'object') return value === 'transcript'
  return false
}

export interface Reference {
  ref: string
  type: string
  label: string
}

/** The references a step at `index` can use for a slot of type `slotType`: the run's input, then each earlier
 * step's outputs of a type the slot accepts (nearest first). */
export function referencesFor(
  steps: PipelineStep[],
  index: number,
  types: Record<string, StepType>,
  slotType: string,
): Reference[] {
  const ids = stepIds(steps)
  const out: Reference[] = []
  for (let i = index - 1; i >= 0; i--) {
    const t = types[steps[i]!.type]
    if (!t) continue
    for (const [name, type] of Object.entries(t.outputs)) {
      if (accepts(slotType, type)) out.push({ ref: `{{steps.${ids[i]}.${name}}}`, type, label: `${ids[i]}.${name}` })
    }
  }
  const inputs: [string, string][] = [
    ['audio', 'audio'],
    ['text', 'text'],
    ['filename', 'text'],
    ['duration', 'number'],
  ]
  for (const [name, type] of inputs) {
    if (accepts(slotType, type)) out.push({ ref: `{{input.${name}}}`, type, label: `input.${name}` })
  }
  return out
}

/** The problems of step `index` (top level), with the path inside the step. */
export function problemsOf(problems: Problem[], index: number): Problem[] {
  const prefix = `steps[${index}]`
  return problems
    .filter((p) => p.path === prefix || p.path.startsWith(prefix + '.'))
    .map((p) => ({ ...p, path: p.path.slice(prefix.length + 1) }))
}

/** Problems not tied to one step (the pipeline's id, outputs, ...). */
export function pipelineProblems(problems: Problem[]): Problem[] {
  return problems.filter((p) => !p.path.startsWith('steps['))
}

/** Move the step at `from` to `to` (both kept in range). */
export function moveStep(steps: PipelineStep[], from: number, to: number): PipelineStep[] {
  if (to < 0 || to >= steps.length || from === to) return steps
  const next = [...steps]
  const [s] = next.splice(from, 1)
  next.splice(to, 0, s!)
  return next
}

/** A definition with empty parts left out, as it is saved and exported. */
export function cleanPipeline(p: Pipeline): Pipeline {
  const steps = p.steps.map((s) => {
    const out: PipelineStep = { ...s }
    if (!out.id) delete out.id
    if (!out.model || out.model === 'auto') delete out.model
    if (out.inputs && Object.keys(out.inputs).length === 0) delete out.inputs
    if (out.options && Object.keys(out.options).length === 0) delete out.options
    if (!out.on_error || out.on_error === 'fail') delete out.on_error
    return out
  })
  const out: Pipeline = { ...p, steps }
  for (const key of ['version', 'created_at', 'updated_at'] as const) delete out[key]
  if (!out.name) delete out.name
  if (!out.description) delete out.description
  if (out.vars && Object.keys(out.vars).length === 0) delete out.vars
  if (out.outputs && Object.keys(out.outputs).length === 0) delete out.outputs
  if (out.triggers && out.triggers.length === 0) delete out.triggers
  if (out.mode === 'file') delete out.mode
  return out
}

/** An option value as the builder's text field shows it, and back. */
export function optionText(value: unknown, type: string): string {
  if (value === undefined || value === null) return ''
  if (type === 'list' && Array.isArray(value) && value.every((v) => typeof v !== 'object'))
    return value.map(String).join(', ')
  if (typeof value === 'object') return JSON.stringify(value, null, 1)
  return String(value)
}

export function parseOption(text: string, type: string): unknown {
  const t = text.trim()
  if (t === '') return undefined
  if (t.includes('{{') && ['number', 'integer', 'boolean'].includes(type)) return t
  switch (type) {
    case 'number':
    case 'integer': {
      const n = Number(t)
      return Number.isFinite(n) ? n : t
    }
    case 'boolean':
      return t === 'true'
    case 'list':
      if (t.startsWith('[')) return tryJson(t)
      return t
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean)
    case 'object':
    case 'steps':
    case 'condition':
    case 'cases':
      return tryJson(t)
    default:
      return text
  }
}

function tryJson(t: string): unknown {
  try {
    return JSON.parse(t)
  } catch {
    return t
  }
}

/** Option types edited as JSON (nested steps, conditions, objects). */
export function isJsonOption(type: string): boolean {
  return ['object', 'steps', 'condition', 'cases'].includes(type)
}

export function isAudio(value: unknown): value is AudioOutput {
  return Boolean(value && typeof value === 'object' && (value as { type?: unknown }).type === 'audio')
}

/** A short text for an output value in the run view. */
export function preview(value: unknown, max = 240): string {
  if (value === null || value === undefined) return '—'
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return text.length > max ? text.slice(0, max - 1) + '…' : text
}

/** Parse an imported definition: JSON, or the simple YAML the server writes is left to the server. */
export function parseImport(text: string): Pipeline | null {
  try {
    const doc = JSON.parse(text) as unknown
    if (doc && typeof doc === 'object' && Array.isArray((doc as Pipeline).steps)) return doc as Pipeline
  } catch {
    return null
  }
  return null
}

/** The answer of `POST /v1/pipelines/simulate`. */
export interface PipelineSimulation {
  object: 'pipeline.simulation'
  turns: { user: string; agent: string; run: PipelineRun }[]
  ended: { action: string; to?: string } | null
  transcript: { role: 'user' | 'assistant'; text: string }[]
  ms: number
}

/** The step types a pipeline of this mode can use (live: only those that run per turn). */
export function typesFor(types: StepType[], mode: string | undefined): StepType[] {
  return mode === 'live' ? types.filter((t) => t.live) : types.filter((t) => !t.live_only)
}
