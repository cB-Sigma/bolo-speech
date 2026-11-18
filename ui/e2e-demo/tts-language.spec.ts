import { expect, test } from '@playwright/test'
import { openDemo, overflow } from './open'

test('text to speech: the voice follows the text’s language, and Kokoro voices blend', async ({ page }) => {
  await openDemo(page, 'tts')
  await expect(page.getByTestId('detect-language')).toBeChecked()
  await page.getByTestId('voice-picker').click()
  await page.getByTestId('voice-kokoro-af-heart').click()
  await page.getByTestId('tts-text').fill('नमस्ते, आप कैसे हैं? मैं ठीक हूँ।')
  await page.getByTestId('speak').click()
  const decision = page.getByTestId('tts-decision')
  await expect(decision).toContainText('detected from the text')
  await expect(page.getByTestId('tts-language-header')).toContainText('source=detected; detected=hi; script=Devanagari')
  await expect(page.getByTestId('voice-picker')).toContainText('Hindi')
  // Without following the text, the voice is kept and the mismatch said.
  await page.getByTestId('voice-picker').click()
  await page.getByTestId('voice-kokoro-af-heart').click()
  await page.getByTestId('tts-text').fill('नमस्ते, आप कैसे हैं? मैं ठीक हूँ।')
  await page.getByTestId('detect-language').click()
  await page.getByTestId('speak').click()
  await expect(page.getByTestId('tts-mismatch')).toContainText('looks like Hindi')
  // A Kokoro voice offers blends, spelled as the server spells them.
  const blend = page.getByTestId('voice-blend')
  await expect(blend).toBeVisible()
  await blend.getByTestId('blend-toggle').click()
  await expect(blend).toContainText(/af_heart\(0\.7\)\+\w+\(0\.3\)/)
  await page.setViewportSize({ width: 390, height: 900 })
  expect(await overflow(page)).toBeLessThanOrEqual(1)
})

test.describe('with Kokoro and the phonemiser available', () => {
  test.skip(
    !process.env.BOLO_E2E_VOICE_DIR || !process.env.BOLO_E2E_MODEL_DIR,
    'set BOLO_E2E_VOICE_DIR (uccharan data) and BOLO_E2E_MODEL_DIR (Kokoro) to run',
  )

  test('a blend of two Kokoro voices is spoken, read in the heaviest voice’s language', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (e) => errors.push(e.message))
    await openDemo(page, 'tts')
    await page.getByTestId('voice-picker').click()
    await page.getByTestId('voice-kokoro-af-heart').click()
    await page.getByTestId('tts-text').fill('Hello there. This voice is a blend of two voices.')
    const blend = page.getByTestId('voice-blend')
    await blend.getByTestId('blend-toggle').click()
    const spelled = await blend.getByTestId('blend-string').innerText()
    await page.getByTestId('speak').click()
    await expect(page.getByTestId('save-wav')).toBeEnabled({ timeout: 170_000 })
    await expect(page.getByTestId('tts-language-header')).toContainText(`voice=${spelled}`)
    console.log(`[blend] ${spelled}: ${(await page.getByTestId('tts-language-header').innerText()).trim()}`)
    expect(errors).toEqual([])
  })
})
