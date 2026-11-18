import { describe, expect, it } from 'vitest'
import { dialogueBody, parseScript, parseSegments, speakersOf, speaksDialogues, toScript } from './dialogue'

describe('dialogue', () => {
  it('reads a script: a name and a colon start a turn, other lines continue it', () => {
    const turns = parseScript('Asha: Hello there.\nRavi: Hi.\nGood to see you.\n\nAsha: Likewise.')
    expect(turns).toEqual([
      { speaker: 'Asha', text: 'Hello there.' },
      { speaker: 'Ravi', text: 'Hi. Good to see you.' },
      { speaker: 'Asha', text: 'Likewise.' },
    ])
    expect(parseScript(toScript(turns))).toEqual(turns)
    expect(speakersOf(turns)).toEqual(['Asha', 'Ravi'])
  })

  it('sends only complete turns and the voices of speakers that are left', () => {
    const body = dialogueBody(
      'kokoro-v1.0',
      [
        { speaker: ' Asha ', text: 'Hello.' },
        { speaker: 'Ravi', text: '  ' },
      ],
      { Asha: 'af_heart', Ravi: 'am_adam' },
      { mode: 'auto', pauseMs: 200, instructions: '  ' },
    )
    expect(body).toEqual({
      model: 'kokoro-v1.0',
      turns: [{ speaker: 'Asha', text: 'Hello.' }],
      voices: { Asha: 'af_heart' },
      mode: undefined,
      pause_ms: 200,
      instructions: undefined,
      response_format: 'wav',
    })
  })

  it('knows the models that speak a dialogue in one take, and reads the segments header', () => {
    expect(speaksDialogues({ features: ['dialogue'] })).toBe(true)
    expect(speaksDialogues({ features: [] })).toBe(false)
    expect(parseSegments('[{"turn":0,"speaker":"A","start":0,"end":1.2}]')).toEqual([
      { turn: 0, speaker: 'A', start: 0, end: 1.2 },
    ])
    expect(parseSegments('not json')).toEqual([])
    expect(parseSegments(null)).toEqual([])
  })
})
