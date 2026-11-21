import type { DecisionAnswer, DecisionQuestion } from '@/lib/api/types'

export type DecisionKind = DecisionQuestion['type']

/** One question as typed in the console: the answers are one per line (options of a choice, levels of a score). */
export interface DraftQuestion {
  type: DecisionKind
  question: string
  answers: string
}

export const KIND_LABEL: Record<DecisionKind, string> = {
  yes_no: 'Yes or no',
  choice: 'One of these',
  score: 'A level on a scale',
}

/** The typed questions as a request's `questions` (ids q1, q2, ...); an error message when one is incomplete. */
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

/** How sure the model is of an answer, 0 to 1 (yes/no: how far the probability is from 0.5). */
export function confidence(a: DecisionAnswer): number {
  if (a.type === 'yes_no') return Math.abs(a.probability - 0.5) * 2
  return a.confidence
}

export type HookAction = 'notify' | 'tag' | 'transfer' | 'end_call'

export const ACTION_LABEL: Record<HookAction, string> = {
  notify: 'Report it',
  tag: 'Tag the call',
  transfer: 'Transfer the call',
  end_call: 'End the call',
}

/** One rule as the editor holds it: when question `question` (index) answers `answer`, do `action`. */
export interface DraftRule {
  question: number
  /** yes_no: 'yes' or 'no'; choice: an option; score: a level; '' for any answer */
  answer: string
  minProbability: string
  action: HookAction
  value: string
}

/** The live-decision hook as the editor holds it (`session.decisions`, a phone agent's `decisions`). */
export interface HookForm {
  enabled: boolean
  model: string
  questions: DraftQuestion[]
  rules: DraftRule[]
}

export const emptyHook = (): HookForm => ({
  enabled: false,
  model: '',
  questions: [{ type: 'yes_no', question: 'Does the caller ask to speak to a person?', answers: '' }],
  rules: [],
})

/** The hook's configuration for the server, null when it is off, or the first thing wrong with it. */
export function hookConfig(form: HookForm): Record<string, unknown> | null | string {
  if (!form.enabled) return null
  const questions = buildQuestions(form.questions)
  if (typeof questions === 'string') return questions
  const actions: Record<string, unknown>[] = []
  for (const [i, r] of form.rules.entries()) {
    const q = questions[`q${r.question + 1}`]
    if (!q) return `Rule ${i + 1} names a question that is not there`
    const rule: Record<string, unknown> = { question: `q${r.question + 1}`, do: r.action }
    const answer = r.answer.trim()
    if (answer) {
      if (q.type === 'yes_no') rule.answer = answer.toLowerCase() === 'yes'
      else if (q.type === 'score' && /^\d+$/.test(answer)) rule.answer = Number(answer)
      else rule.answer = answer
    }
    if (r.minProbability.trim()) {
      const p = Number(r.minProbability)
      if (!Number.isFinite(p) || p < 0 || p > 1) return `Rule ${i + 1}: the probability must be between 0 and 1`
      rule.min_probability = p
    }
    if (r.action === 'tag' && !r.value.trim()) return `Rule ${i + 1}: a tag rule needs the tag`
    if (r.value.trim()) rule.value = r.value.trim()
    actions.push(rule)
  }
  return { ...(form.model ? { model: form.model } : {}), questions, actions }
}

interface ServerHook {
  model?: string | null
  questions?: Record<string, { type?: string; question?: string; options?: unknown; levels?: unknown }>
  actions?: { question?: string; answer?: unknown; min_probability?: number; do?: string; value?: string }[]
}

/** The editor's form for a stored hook (a phone agent's `decisions`); off when there is none. */
export function hookForm(hook: ServerHook | null | undefined): HookForm {
  if (!hook?.questions) return emptyHook()
  const ids = Object.keys(hook.questions)
  const questions: DraftQuestion[] = ids.map((id) => {
    const q = hook.questions?.[id] ?? {}
    const list = q.type === 'choice' ? q.options : q.levels
    const lines = Array.isArray(list) ? list.map(String) : list && typeof list === 'object' ? Object.keys(list) : []
    const type: DecisionKind = q.type === 'choice' || q.type === 'score' ? q.type : 'yes_no'
    return { type, question: String(q.question ?? ''), answers: lines.join('\n') }
  })
  const rules: DraftRule[] = (hook.actions ?? []).map((a) => ({
    question: Math.max(0, ids.indexOf(String(a.question))),
    answer: a.answer === true ? 'yes' : a.answer === false ? 'no' : a.answer == null ? '' : String(a.answer),
    minProbability: a.min_probability == null ? '' : String(a.min_probability),
    action: (['notify', 'tag', 'transfer', 'end_call'].includes(String(a.do)) ? a.do : 'notify') as HookAction,
    value: a.value ?? '',
  }))
  return { enabled: true, model: hook.model ?? '', questions, rules }
}

/** One `conversation.item.decisions.completed` the console shows. */
export interface LiveDecision {
  itemId: string
  model: string
  answers: Record<string, DecisionAnswer>
  actions: { question: string; do: string; value?: string }[]
}
