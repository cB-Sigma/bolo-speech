import { describe, expect, it } from 'vitest'
import { intentLabel, slotRows } from './intent-panels'

describe('spoken intent', () => {
  it('names the intent', () => {
    expect(intentLabel('alarm_set')).toBe('alarm set')
    expect(intentLabel(null)).toBe('not understood')
  })

  it('lists the slots in the model order', () => {
    expect(
      slotRows({
        object: 'audio.intent',
        model: 'slu-slurp-hubert',
        duration: 2,
        intent: 'calendar_set',
        scenario: 'calendar',
        action: 'set',
        slots: [
          { type: 'event_name', value: 'dentist' },
          { type: 'date', value: 'tomorrow' },
        ],
        text: '',
      }),
    ).toEqual([
      ['event name', 'dentist'],
      ['date', 'tomorrow'],
    ])
  })
})
