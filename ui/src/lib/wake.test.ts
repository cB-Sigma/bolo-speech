import { describe, expect, it } from 'vitest'
import { applyWakeMessage, idleWake, splitKeywords } from './wake'

describe('wake words', () => {
  it('splits typed phrases', () => {
    expect(splitKeywords('hey computer\nlights on, ok bolo;')).toEqual(['hey computer', 'lights on', 'ok bolo'])
    expect(splitKeywords('  ')).toEqual([])
  })

  it('follows the stream: ready, detections newest first, errors', () => {
    let s = applyWakeMessage(idleWake, { type: 'ready', model: 'kws', keywords: ['hey jarvis'] })
    expect(s).toMatchObject({ status: 'listening', model: 'kws', keywords: ['hey jarvis'] })
    s = applyWakeMessage(s, { type: 'wake', keyword: 'hey jarvis', time: 1.8 }, 1)
    s = applyWakeMessage(s, { type: 'wake', keyword: 'hey jarvis', time: 5.2, score: 0.9 }, 2)
    expect(s.hits.map((h) => h.time)).toEqual([5.2, 1.8])
    expect(s.hits[0]?.score).toBe(0.9)
    s = applyWakeMessage(s, { type: 'error', message: 'no such model' })
    expect(s).toMatchObject({ status: 'error', error: 'no such model' })
  })
})
