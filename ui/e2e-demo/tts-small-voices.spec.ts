import { expect, test } from '@playwright/test'
import { openDemo } from './open'

/**
 * little-tts, Paradee-8M, Kitten TTS 2 Pico and kale-tts-1-nano in the page, their files from Hugging Face. Paradee
 * and Kitten read uccharan's Kokoro phonemes, whose data the demo serves next to its voices (BOLO_E2E_VOICE_DIR).
 */
test.describe('small voices with their files from the internet', () => {
  test.skip(
    !process.env.BOLO_E2E_NETWORK || !process.env.BOLO_E2E_VOICE_DIR,
    'set BOLO_E2E_NETWORK=1 and BOLO_E2E_VOICE_DIR to fetch the voices from Hugging Face',
  )

  const cases: [string, string][] = [
    ['little-tts-92', 'The weather is nice today, and the meeting moved to 3:30 pm.'],
    ['paradee-af-heart', 'The weather is nice today, and the meeting moved to half past three.'],
    ['paradee-bf-isabella', 'I took a bath before the meeting at half past three.'],
    ['kitten-pico-en-eleanor', 'The weather is nice today, and the meeting moved to half past three.'],
    ['kitten-pico-hi-priya', 'नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।'],
    ['kale-kenzie', 'The weather is nice today, and the meeting moved to 3:30 pm.'],
    ['kale-jack', 'The weather is nice today, and the meeting moved to 3:30 pm.'],
  ]
  for (const [voice, text] of cases) {
    test(`${voice} speaks a sentence`, async ({ page }) => {
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(e.message))
      await openDemo(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId(`voice-${voice}`).click()
      await page.getByTestId('tts-text').fill(text)
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('save-wav')).toBeEnabled({ timeout: 170_000 })
      const download = page.waitForEvent('download')
      await page.getByTestId('save-wav').click()
      const chunks: Buffer[] = []
      for await (const c of await (await download).createReadStream()) chunks.push(c as Buffer)
      const wav = Buffer.concat(chunks)
      const rate = wav.readUInt32LE(24)
      const pcm = new Int16Array(wav.buffer.slice(wav.byteOffset + 44, wav.byteOffset + wav.length))
      let sum = 0
      for (const s of pcm) sum += (s / 32768) ** 2
      expect(rate).toBe(24000)
      expect(pcm.length / rate).toBeGreaterThan(1.5)
      expect(Math.sqrt(sum / pcm.length)).toBeGreaterThan(0.02)
      expect(errors).toEqual([])
    })
  }
})
