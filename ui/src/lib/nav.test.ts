import { describe, expect, it } from 'vitest'
import { visibleNav } from './nav'

describe('visibleNav', () => {
  it('offers Speakers only on servers that enrol speakers, and Analyse audio everywhere', () => {
    const without = visibleNav({ wordLists: false }).map((n) => n.to)
    expect(without).toContain('/analyse')
    expect(without).not.toContain('/speakers')
    const withSpeakers = visibleNav({ wordLists: false, speakers: true }).filter((n) => n.to === '/speakers')
    expect(withSpeakers).toHaveLength(1)
    expect(withSpeakers[0]?.description).toMatch(/verify or identify/)
  })
})
