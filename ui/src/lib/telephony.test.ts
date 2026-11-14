import { describe, expect, it } from 'vitest'
import { blankAgent } from '@/mock/telephony'
import type { CallTurn } from './api/types'
import { agentForm, agentUpdate, latencyShares, outcomeLabel, providerState } from './telephony'

describe('agent form', () => {
  it('round-trips an agent and sends unset text as null', () => {
    const agent = { ...blankAgent('desk'), greeting: 'Hello.', tools: [{ name: 'lookup', url: 'https://x.example/l' }] }
    const { body, problem } = agentUpdate(agentForm(agent))
    expect(problem).toBeNull()
    expect(body).toMatchObject({ greeting: 'Hello.', instructions: null, max_duration_s: 600, silence_prompts: 2 })
    expect(body.tools).toEqual([{ name: 'lookup', url: 'https://x.example/l' }])
  })

  it('names what is wrong', () => {
    const form = agentForm(blankAgent('desk'))
    expect(agentUpdate({ ...form, maxDurationS: 'long' }).problem).toMatch(/longest call length/)
    expect(agentUpdate({ ...form, transferNumber: 'reception' }).problem).toMatch(/international form/)
    expect(agentUpdate({ ...form, voicemail: 'leave_message' }).problem).toMatch(/message to leave/)
    expect(agentUpdate({ ...form, tools: '{"name": "x"}' }).problem).toMatch(/JSON list/)
    expect(agentUpdate({ ...form, tools: '[{' }).problem).toMatch(/not valid JSON/)
  })
})

describe('call helpers', () => {
  it('labels outcomes and states', () => {
    expect(outcomeLabel('no_response')).toBe('Caller went silent')
    expect(outcomeLabel(null, 'in_progress')).toBe('In progress')
    expect(outcomeLabel('something_new')).toBe('something_new')
    const p = { exposed: false, configured: true } as Parameters<typeof providerState>[0]
    expect(providerState(p)).toEqual({ label: 'Closed: settings missing', ready: false })
    expect(providerState({ ...p, exposed: true }).ready).toBe(true)
  })

  it('splits a turn into the measured parts', () => {
    const latency = {
      endpoint_ms: 500,
      stt_ms: 100,
      llm_first_token_ms: 300,
      tts_first_audio_ms: 100,
      send_ms: null,
      total_ms: 500,
      response_ms: 1000,
    }
    const shares = latencyShares({ latency } as CallTurn)
    expect(shares.map((s) => s.key)).toEqual(['endpoint_ms', 'stt_ms', 'llm_first_token_ms', 'tts_first_audio_ms'])
    expect(shares[0]?.share).toBeCloseTo(0.5)
    expect(shares.reduce((sum, s) => sum + s.share, 0)).toBeCloseTo(1)
  })
})
