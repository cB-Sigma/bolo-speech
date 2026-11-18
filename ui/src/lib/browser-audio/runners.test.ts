import { describe, expect, it } from 'vitest'
import { readAudioSetLabels } from './runners'

describe('readAudioSetLabels', () => {
  it('reads display names by index, with commas inside quotes', () => {
    const csv = 'index,mid,display_name\n0,/m/09x0r,"Speech"\n2,/m/05zppz,"Male speech, man speaking"\n1,/m/02zsn,Female\n'
    expect(readAudioSetLabels(csv)).toEqual(['Speech', 'Female', 'Male speech, man speaking'])
  })
})
