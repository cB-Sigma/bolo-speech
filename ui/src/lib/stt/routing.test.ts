import { describe, expect, it } from 'vitest'
import { browserModel } from '@/lib/browser-stt/models'
import { routeByLanguage } from './routing'

const meta = { seconds: 4, ms: 120, lid: 'voxlingua107-ecapa-int8' }

describe('routing by the language heard (the server’s model=auto)', () => {
  it('keeps a recogniser that speaks the language, and tells a multilingual one', () => {
    expect(routeByLanguage([['en', 0.98]], browserModel('moonshine-tiny'), () => false, meta)).toMatchObject({
      reason: 'kept',
      model: 'moonshine-tiny',
      language: null,
    })
    expect(routeByLanguage([['hi', 0.9]], browserModel('whisper-base-q8'), () => false, meta)).toMatchObject({
      reason: 'kept',
      language: 'hi',
    })
  })

  it('sends the clip to a downloaded recogniser made for the language', () => {
    const d = routeByLanguage([['hi', 0.97]], browserModel('moonshine-tiny'), (id) => id === 'whisper-tiny-hindi', meta)
    expect(d).toMatchObject({ reason: 'routed', model: 'whisper-tiny-hindi' })
    expect(d.header).toBe(
      'auto -> whisper-tiny-hindi; reason=routed; language=hi; probability=0.97; lid=voxlingua107-ecapa-int8',
    )
  })

  it('only suggests a recogniser that is not downloaded, and trusts nothing below 0.5', () => {
    const d = routeByLanguage([['hi', 0.97]], browserModel('moonshine-tiny'), () => false, meta)
    expect(d).toMatchObject({ reason: 'recommended', model: 'moonshine-tiny', recommended: 'whisper-tiny-hindi' })
    expect(d.header).toContain('recommended=whisper-tiny-hindi')
    expect(routeByLanguage([['hi', 0.4]], browserModel('moonshine-tiny'), () => true, meta).reason).toBe('unsure')
  })
})
