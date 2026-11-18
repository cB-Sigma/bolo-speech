import { describe, expect, it } from 'vitest'
import { encodeWav } from './audio/wav'
import {
  parseTranscription,
  separationSummary,
  speakerCountProblem,
  speakersOf,
  talkTime,
  transcriptionForm,
  wavSampleRate,
} from './diarize'

const blob = new Blob(['x'])

describe('transcriptionForm', () => {
  it('asks for diarized_json with the diarizer, speaker bounds and separation', () => {
    const f = transcriptionForm(blob, 'a.wav', {
      model: 'whisper-base',
      language: 'en',
      diarize: true,
      diarizer: 'nemotron-3-diarization',
      speakers: { mode: 'range', min: 2, max: 4 },
      separate: true,
      separationModel: 'voxlock-int8',
      allowNarrowband: true,
    })
    expect(Object.fromEntries([...f.entries()].filter(([k]) => k !== 'file'))).toEqual({
      model: 'whisper-base',
      language: 'en',
      response_format: 'diarized_json',
      diarize: 'true',
      diarization_model: 'nemotron-3-diarization',
      min_speakers: '2',
      max_speakers: '4',
      separate_overlaps: 'true',
      separation_model: 'voxlock-int8',
      separation_allow_narrowband: 'true',
    })
  })
  it('sends an exact count, and nothing diarization-related without diarization', () => {
    expect(
      transcriptionForm(blob, 'a', { model: 'm', diarize: true, speakers: { mode: 'exact', n: 3 } }).get(
        'num_speakers',
      ),
    ).toBe('3')
    const plain = transcriptionForm(blob, 'a', { model: 'm', diarize: false, language: 'auto' })
    expect(plain.get('response_format')).toBe('verbose_json')
    expect(plain.has('diarize')).toBe(false)
    expect(plain.has('language')).toBe(false)
  })
  it('sends hotwords and punctuation only when they are set', () => {
    const f = transcriptionForm(blob, 'a', {
      model: 'm',
      diarize: false,
      hotwords: ' Zintrava Labs, Wadhwani ',
      punctuation: 'auto',
    })
    expect(f.get('hotwords')).toBe('Zintrava Labs, Wadhwani')
    expect(f.get('punctuation')).toBe('auto')
    const plain = transcriptionForm(blob, 'a', { model: 'm', diarize: false, hotwords: '  ', punctuation: '' })
    expect(plain.has('hotwords')).toBe(false)
    expect(plain.has('punctuation')).toBe(false)
  })
  it('refuses separation without speaker labels', () => {
    expect(() => transcriptionForm(blob, 'a', { model: 'm', diarize: false, separate: true })).toThrow(/diarization/)
  })
})

describe('speaker counts', () => {
  it('validates hints', () => {
    expect(speakerCountProblem({ mode: 'auto' })).toBeNull()
    expect(speakerCountProblem({ mode: 'exact', n: 0 })).not.toBeNull()
    expect(speakerCountProblem({ mode: 'range', min: 3, max: 2 })).toMatch(/cannot be more/)
    expect(speakerCountProblem({ mode: 'range', max: 2 })).toBeNull()
  })
})

describe('parseTranscription', () => {
  const body = {
    text: 'hi there',
    duration: 9,
    segments: [
      {
        type: 'transcript.text.segment',
        id: 'seg_1',
        start: 3.5,
        end: 7,
        text: ' there ',
        speaker: 'B',
        overlap: true,
        separated: true,
      },
      { id: 'seg_0', start: 0, end: 5, text: 'hi', speaker: 'A', overlap: true },
    ],
    separation: {
      model: 'voxlock-int8',
      status: 'applied',
      reason: null,
      detail: null,
      windows: [{ start: 1, end: 8, separated: ['A', 'B'] }],
    },
  }
  it('normalises segments and reads the separation header', () => {
    const r = parseTranscription(body, new Headers({ 'x-bolo-separation': 'applied', 'x-request-id': 'r_1' }))
    expect(r.segments[0]).toMatchObject({ text: 'there', speaker: 'B', overlap: true, separated: true })
    expect(r.segments[1]).toMatchObject({ separated: false })
    expect(r.separationHeader).toBe('applied')
    expect(r.requestId).toBe('r_1')
    expect(speakersOf(r.segments)).toEqual(['A', 'B'])
    expect(talkTime(r.segments).get('A')).toBe(5)
    expect(separationSummary(r.separation!)).toBe('voxlock-int8 separated the voices in 1 window.')
  })
  it('explains skipped separation', () => {
    expect(
      separationSummary({ model: 'v', status: 'skipped', reason: 'no_overlap', detail: null, windows: [] }),
    ).toMatch(/No overlapping speech/)
    expect(
      separationSummary({ model: 'v', status: 'partial', reason: 'timeout', detail: 'took too long', windows: [] }),
    ).toBe('v separated some of the voices. Took too long.')
  })
})

describe('wavSampleRate', () => {
  it('reads WAV headers and ignores anything else', () => {
    const wav = encodeWav(new Float32Array(100), 8000)
    expect(wavSampleRate(wav.slice().buffer)).toBe(8000)
    expect(wavSampleRate(new Uint8Array(64).buffer)).toBeNull()
  })
})
