import { Plus, Trash2 } from 'lucide-react'
import { CatalogSelect } from '@/components/catalog-select'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { useInfo, useModels } from '@/hooks/use-api'
import { autoPull } from '@/lib/capabilities'
import {
  ACTION_LABEL,
  KIND_LABEL,
  type DecisionKind,
  type DraftRule,
  type HookAction,
  type HookForm,
} from '@/lib/decisions'

/**
 * Live decisions: questions a decision model answers after every user turn, and rules that act on the answers
 * (`session.decisions` of a conversation, `decisions` of a phone agent). `call` offers the actions that move a call.
 */
export function HookEditor({
  value,
  onChange,
  disabled,
  call = false,
  testId = 'decision-hook',
}: {
  value: HookForm
  onChange: (v: HookForm) => void
  disabled?: boolean
  call?: boolean
  testId?: string
}) {
  const info = useInfo()
  const models = useModels('decision').models
  const defaultId = info.data?.capabilities?.decision?.default_model ?? null
  const set = (patch: Partial<HookForm>) => onChange({ ...value, ...patch })
  const setQuestion = (i: number, patch: Partial<HookForm['questions'][number]>) =>
    set({ questions: value.questions.map((q, j) => (j === i ? { ...q, ...patch } : q)) })
  const setRule = (i: number, patch: Partial<DraftRule>) =>
    set({ rules: value.rules.map((r, j) => (j === i ? { ...r, ...patch } : r)) })
  const actions: HookAction[] = call ? ['notify', 'tag', 'transfer', 'end_call'] : ['notify', 'tag']

  return (
    <div className="grid gap-2.5" data-testid={testId}>
      <label className="flex items-center gap-2 text-[13px]">
        <Switch
          checked={value.enabled}
          onCheckedChange={(enabled) => set({ enabled })}
          disabled={disabled}
          aria-label="Ask decision questions after every turn"
        />
        Ask questions after every {call ? 'caller' : 'user'} turn
      </label>
      {value.enabled && (
        <>
          <CatalogSelect
            models={models}
            value={value.model}
            onChange={(model) => set({ model })}
            label="Decision model"
            defaultLabel={defaultId ? `Server default (${defaultId})` : 'Server default'}
            disabled={disabled}
            autoPull={autoPull(info.data)}
            testId={`${testId}-model`}
          />
          {value.questions.map((q, i) => (
            <div key={i} className="grid gap-1.5 rounded-md border border-border p-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">q{i + 1}</span>
                <Input
                  value={q.question}
                  onChange={(e) => setQuestion(i, { question: e.target.value })}
                  placeholder="The question"
                  aria-label={`Question ${i + 1}`}
                  disabled={disabled}
                />
                <Select
                  value={q.type}
                  onValueChange={(v) => setQuestion(i, { type: v as DecisionKind })}
                  disabled={disabled}
                >
                  <SelectTrigger className="h-9 w-40 shrink-0" aria-label={`Answer type of question ${i + 1}`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent position="popper" align="end">
                    {(Object.keys(KIND_LABEL) as DecisionKind[]).map((k) => (
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
                  disabled={disabled || value.questions.length === 1}
                  onClick={() =>
                    set({
                      questions: value.questions.filter((_, j) => j !== i),
                      rules: value.rules
                        .filter((r) => r.question !== i)
                        .map((r) => (r.question > i ? { ...r, question: r.question - 1 } : r)),
                    })
                  }
                >
                  <Trash2 />
                </Button>
              </div>
              {q.type !== 'yes_no' && (
                <Textarea
                  value={q.answers}
                  onChange={(e) => setQuestion(i, { answers: e.target.value })}
                  placeholder={q.type === 'choice' ? 'One option per line' : 'One level per line, lowest first'}
                  aria-label={`${q.type === 'choice' ? 'Options' : 'Levels'} of question ${i + 1}`}
                  rows={2}
                  disabled={disabled}
                />
              )}
            </div>
          ))}
          {value.rules.map((r, i) => (
            <div
              key={i}
              className="flex flex-wrap items-center gap-1.5 text-[13px]"
              data-testid={`${testId}-rule${i + 1}`}
            >
              <span className="text-muted-foreground">When</span>
              <Select
                value={String(r.question)}
                onValueChange={(v) => setRule(i, { question: Number(v) })}
                disabled={disabled}
              >
                <SelectTrigger className="h-8 w-20" aria-label={`Question of rule ${i + 1}`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent position="popper">
                  {value.questions.map((_, j) => (
                    <SelectItem key={j} value={String(j)}>
                      q{j + 1}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-muted-foreground">answers</span>
              <Input
                className="h-8 w-28"
                value={r.answer}
                onChange={(e) => setRule(i, { answer: e.target.value })}
                placeholder={value.questions[r.question]?.type === 'yes_no' ? 'yes or no' : 'any'}
                aria-label={`Answer of rule ${i + 1}`}
                disabled={disabled}
              />
              <span className="text-muted-foreground">with probability ≥</span>
              <Input
                className="h-8 w-16"
                value={r.minProbability}
                onChange={(e) => setRule(i, { minProbability: e.target.value })}
                placeholder="0.8"
                aria-label={`Least probability of rule ${i + 1}`}
                disabled={disabled}
              />
              <Select
                value={r.action}
                onValueChange={(v) => setRule(i, { action: v as HookAction })}
                disabled={disabled}
              >
                <SelectTrigger className="h-8 w-40" aria-label={`Action of rule ${i + 1}`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent position="popper">
                  {actions.map((a) => (
                    <SelectItem key={a} value={a}>
                      {ACTION_LABEL[a]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {(r.action === 'tag' || r.action === 'transfer') && (
                <Input
                  className="h-8 w-36"
                  value={r.value}
                  onChange={(e) => setRule(i, { value: e.target.value })}
                  placeholder={r.action === 'tag' ? 'tag' : 'number (agent default)'}
                  aria-label={`Value of rule ${i + 1}`}
                  disabled={disabled}
                />
              )}
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove rule ${i + 1}`}
                disabled={disabled}
                onClick={() => set({ rules: value.rules.filter((_, j) => j !== i) })}
              >
                <Trash2 />
              </Button>
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={disabled || value.questions.length >= 32}
              onClick={() => set({ questions: [...value.questions, { type: 'yes_no', question: '', answers: '' }] })}
            >
              <Plus /> Question
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={disabled}
              onClick={() =>
                set({
                  rules: [...value.rules, { question: 0, answer: '', minProbability: '', action: 'notify', value: '' }],
                })
              }
            >
              <Plus /> Rule
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
