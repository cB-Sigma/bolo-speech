import { describe, expect, it } from 'vitest'
import { visibleNav } from './nav'
import {
  accepts,
  cleanPipeline,
  moveStep,
  optionText,
  parseOption,
  problemsOf,
  referencesFor,
  stepIds,
  type StepType,
} from './pipelines'

const type = (name: string, outputs: Record<string, string>): StepType => ({
  type: name,
  kind: name,
  group: 'model',
  help: '',
  inputs: [],
  options: [],
  outputs,
  live: false,
  live_only: false,
})

const TYPES: Record<string, StepType> = {
  denoise: type('denoise', { audio: 'audio' }),
  stt: type('stt', { text: 'text', transcript: 'transcript', segments: 'list' }),
  'split-speakers': type('split-speakers', { speakers: 'list' }),
}

describe('pipelines', () => {
  it('numbers steps without ids as the server does', () => {
    expect(stepIds([{ type: 'stt' }, { type: 'stt' }, { type: 'split-speakers' }, { id: 'stt_3', type: 'x' }])).toEqual(
      ['stt', 'stt_2', 'split_speakers', 'stt_3'],
    )
  })

  it('offers only earlier outputs of a type the slot takes, nearest first', () => {
    const steps = [{ type: 'denoise' }, { type: 'stt' }, { type: 'stt' }]
    const text = referencesFor(steps, 2, TYPES, 'text').map((r) => r.label)
    expect(text).toEqual(['stt.text', 'stt.transcript', 'input.text', 'input.filename', 'input.duration'])
    const audio = referencesFor(steps, 2, TYPES, 'audio').map((r) => r.ref)
    expect(audio).toEqual(['{{steps.denoise.audio}}', '{{input.audio}}'])
    expect(accepts('audio', 'transcript')).toBe(false)
    expect(accepts('list', 'turns')).toBe(true)
  })

  it('places problems on their step and edits options', () => {
    const problems = [
      { path: 'steps[1].model', message: 'unknown' },
      { path: 'steps[10].type', message: 'x' },
      { path: 'id', message: 'bad' },
    ]
    expect(problemsOf(problems, 1)).toEqual([{ path: 'model', message: 'unknown' }])
    expect(moveStep([{ type: 'a' }, { type: 'b' }], 1, 0).map((s) => s.type)).toEqual(['b', 'a'])
    expect(parseOption('yes, no', 'list')).toEqual(['yes', 'no'])
    expect(parseOption('0.5', 'number')).toBe(0.5)
    expect(parseOption('{{vars.n}}', 'integer')).toBe('{{vars.n}}')
    expect(parseOption('{"value": "x"', 'condition')).toBe('{"value": "x"') // kept as typed until it parses
    expect(optionText(['a', 'b'], 'list')).toBe('a, b')
    expect(cleanPipeline({ id: 'p', version: 3, steps: [{ type: 'stt', model: 'auto', options: {} }] })).toEqual({
      id: 'p',
      steps: [{ type: 'stt' }],
    })
  })

  it('shows the Pipelines screen on servers that have pipelines', () => {
    expect(visibleNav({ wordLists: false }).map((n) => n.to)).not.toContain('/pipelines')
    expect(visibleNav({ wordLists: false, pipelines: true }).map((n) => n.to)).toContain('/pipelines')
  })
})
