import { describe, expect, it } from 'vitest'
import { emptyTranscript, type TranscriptState } from '@/lib/stt/transcriber'
import { Conversation, takeSentences, type Speaker } from './conversation'
import type { ChatBackend } from './engine'

const final = (text: string) => ({ text, t0: 0, t1: 1, words: null, language: 'en', latencyMs: 120 })

function fakes(reply: string, tokenDelay = 1) {
  const said: string[] = []
  let stopped = 0
  const speaker: Speaker = {
    async say(s) {
      said.push(s)
    },
    stop: () => void (stopped += 1),
    drained: async () => undefined,
  }
  const llm: ChatBackend = {
    async reply(_m, onToken, signal) {
      for (const w of reply.split(/(?<= )/)) {
        if (signal.aborted) break
        await new Promise((r) => setTimeout(r, tokenDelay))
        onToken(w)
      }
      return reply
    },
  }
  return { said, speaker, llm, stopped: () => stopped }
}

describe('voice conversation in the page', () => {
  it('cuts a streaming reply into sentences', () => {
    expect(takeSentences('Hello there. How are')).toEqual({ sentences: ['Hello there.'], rest: 'How are' })
    expect(takeSentences('नमस्ते। आप कैसे हैं', true).sentences).toEqual(['नमस्ते।', 'आप कैसे हैं'])
  })

  it('answers an end of turn sentence by sentence, with latencies', async () => {
    const f = fakes('Sure. It is sunny today. ')
    const c = new Conversation(() => f.llm, f.speaker)
    const t: TranscriptState = { ...emptyTranscript(), finals: [final('What is the weather?')] }
    c.onTranscript(t)
    await new Promise((r) => setTimeout(r, 60))
    expect(f.said).toEqual(['Sure.', 'It is sunny today.'])
    const turns = c.store.get().turns
    expect(turns.map((x) => x.role)).toEqual(['user', 'assistant'])
    expect(turns[1]?.latency?.sttMs).toBe(120)
    expect(turns[1]?.latency?.firstTokenMs).toBeGreaterThanOrEqual(0)
    expect(turns[1]?.latency?.firstAudioMs).toBeGreaterThanOrEqual(0)
    expect(c.store.get().status).toBe('listening')
  })

  it('barge-in: speaking while it answers stops the reply and the speech', async () => {
    const f = fakes('One. Two. Three. Four. Five. Six. ', 20)
    const c = new Conversation(() => f.llm, f.speaker)
    c.onTranscript({ ...emptyTranscript(), finals: [final('Count to six.')] })
    await new Promise((r) => setTimeout(r, 50))
    c.onTranscript({ ...emptyTranscript(), finals: [final('Count to six.')], speech: [{ t0: 2, t1: null }] })
    await new Promise((r) => setTimeout(r, 150))
    expect(f.stopped()).toBe(1)
    expect(c.store.get().bargeIns).toBe(1)
    expect(c.store.get().turns.at(-1)?.interrupted).toBe(true)
    expect(f.said.length).toBeLessThan(6)
  })
})
