import { describe, expect, it } from 'vitest'
import { parseLanguageHeader } from '@/lib/blend'
import { voiceForText } from './language'
import { browserVoice, BROWSER_VOICES } from './voices'

describe('the voice for a text, as the server decides it', () => {
  const english = browserVoice('kokoro-af-heart')

  it('keeps a voice that speaks the text’s language', () => {
    const c = voiceForText('Hello, how are you doing today? It is a good day.', english, BROWSER_VOICES, {
      follow: true,
    })
    expect(c.switched).toBe(false)
    expect(c.header).toBe('en-US; source=detected; detected=en; script=Latin; voice=kokoro-af-heart')
  })

  it('switches to a voice of the text’s language, of the same kind and gender first', () => {
    const c = voiceForText('नमस्ते, आप कैसे हैं? मैं ठीक हूँ।', english, BROWSER_VOICES, { follow: true })
    expect(c.switched).toBe(true)
    expect(c.voice.id).toBe('kokoro-hf-alpha')
    expect(parseLanguageHeader(c.header)).toMatchObject({ source: 'detected', detected: 'hi', script: 'Devanagari' })
  })

  it('prefers a downloaded voice, and falls back to the closest language of the script', () => {
    const syspin = BROWSER_VOICES.find((v) => v.language === 'mr' && v.kind === 'coqui')
    expect(syspin).toBeDefined()
    const c = voiceForText('मी घरी आहे आणि तुम्ही कुठे आहात?', english, BROWSER_VOICES, {
      follow: true,
      downloaded: (v) => v.id === syspin?.id,
    })
    expect(c.voice.id).toBe(syspin?.id)
    expect(c.header).toContain('detected=mr')
  })

  it('reports a mismatch instead of switching when told not to, or for a blend', () => {
    const c = voiceForText('नमस्ते, आप कैसे हैं?', english, BROWSER_VOICES, { follow: false })
    expect(c.switched).toBe(false)
    expect(parseLanguageHeader(c.header)?.mismatch).toBe(true)
    const b = voiceForText('नमस्ते, आप कैसे हैं?', english, BROWSER_VOICES, {
      follow: true,
      blend: 'af_heart(0.7)+af_bella(0.3)',
    })
    expect(b.header).toContain('voice=af_heart(0.7)+af_bella(0.3)')
    expect(b.switched).toBe(false)
  })

  it('leaves an unsure text to the voice', () => {
    const c = voiceForText('OK', english, BROWSER_VOICES, { follow: true })
    expect(c.header).toContain('detected=unsure')
    expect(c.switched).toBe(false)
  })
})
