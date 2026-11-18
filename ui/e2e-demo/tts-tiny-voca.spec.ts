import { expect, test } from '@playwright/test'
import { openDemo } from './open'

test.describe('tiny-voca with its phonemiser from the internet', () => {
  test.skip(
    !process.env.BOLO_E2E_NETWORK,
    'set BOLO_E2E_NETWORK=1 to fetch tiny-voca and its phonemiser data from Hugging Face',
  )

  for (const voice of ['tiny-voca-f2', 'tiny-voca-m2']) {
    test(`${voice} speaks a sentence with a time in it`, async ({ page }) => {
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(e.message))
      await openDemo(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId(`voice-${voice}`).click()
      await page.getByTestId('tts-text').fill('The weather is nice today, and the meeting moved to 3:30 pm.')
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('save-wav')).toBeEnabled({ timeout: 170_000 })
      const download = page.waitForEvent('download')
      await page.getByTestId('save-wav').click()
      const wav = await (await download).createReadStream()
      const chunks: Buffer[] = []
      for await (const c of wav) chunks.push(c as Buffer)
      const pcm = new Int16Array(Buffer.concat(chunks).buffer.slice(44))
      let sum = 0
      for (const s of pcm) sum += (s / 32768) ** 2
      expect(pcm.length / 24000).toBeGreaterThan(1.5)
      expect(Math.sqrt(sum / pcm.length)).toBeGreaterThan(0.02)
      expect(errors).toEqual([])
    })
  }
})
