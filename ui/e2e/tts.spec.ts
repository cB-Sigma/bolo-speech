import { expect, test } from '@playwright/test'
import { open, watchErrors } from './helpers'

test.use({ viewport: { width: 1440, height: 900 } })

test('streams speech over the websocket with timing readouts and a chunk timeline', async ({ page }) => {
  const errors = watchErrors(page)
  await open(page, '/tts')
  await page.getByTestId('speak').click()
  await expect(page.getByTestId('ttfa')).toHaveText(/^\d+$/, { timeout: 10_000 })
  await expect(page.getByTestId('chunk-timeline').getByRole('listitem').first()).toBeVisible()
  await expect(page.getByTestId('underruns')).toContainText(/No underruns|underrun/)
  // Playback finishes on its own; the button returns to Speak.
  await expect(page.getByTestId('speak')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('chunk-timeline').getByRole('listitem')).toHaveCount(5)
  await expect(page.getByTestId('readout').filter({ hasText: 'Real-time factor' })).toContainText(/0\.\d+/)
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'WAV' }).click()
  expect((await download).suggestedFilename()).toMatch(/^bolo-.*\.wav$/)
  expect(errors).toEqual([])
})

test('Hindi sample text and the chunked HTTP transport', async ({ page }) => {
  await open(page, '/tts')
  await page
    .getByRole('button', { name: /^Hindi/ })
    .first()
    .click()
  await expect(page.getByLabel('Text to speak')).toHaveValue(/नमस्ते/)
  await page.getByRole('radio', { name: 'Chunked HTTP' }).click()
  await page.getByTestId('speak').click()
  await expect(page.getByTestId('ttfa')).toHaveText(/^\d+$/, { timeout: 10_000 })
  await page.getByTestId('tts-stop').click()
  await expect(page.getByTestId('speak')).toBeVisible()
})

test('speak, then transcribe reports a word error rate with a diff', async ({ page }) => {
  await open(page, '/tts')
  await page.getByTestId('round-trip').click()
  await expect(page.getByTestId('round-trip-panel')).toBeVisible()
  await expect(page.getByTestId('wer')).toHaveText(/^\d+(\.\d)?%$/, { timeout: 30_000 })
  await expect(page.getByTestId('wer-diff')).toContainText(/thanks for calling/i)
})

test('voice picker lists the model voices and previews one', async ({ page }) => {
  await open(page, '/tts', { storage: { 'bolo.prefs': JSON.stringify({ ttsModel: 'kokoro-82m' }) } })
  await page.getByRole('combobox', { name: 'Voice' }).click()
  // The model's seven voices, and "From the text" (no voice named: the server picks one for the detected language).
  await expect(page.getByRole('option')).toHaveCount(8)
  await expect(page.getByRole('option', { name: /From the text/ })).toBeVisible()
  await page.getByRole('option', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: 'Preview this voice' }).click()
  await expect(page.getByRole('combobox', { name: 'Voice' })).toContainText('Alpha')
})

test('an "In browser" mode runs the same screen with voices that run in the page', async ({ page }) => {
  // The voices normally come from the public demo; point at a host that answers 404 so nothing is fetched.
  await page.addInitScript(() =>
    localStorage.setItem('bolo.browserTts', JSON.stringify({ voicesBase: '/no-voices-here/' })),
  )
  await open(page, '/tts')
  await expect(page.getByTestId('tts-mode-server')).toHaveAttribute('data-state', 'on')
  await page.getByTestId('tts-mode-browser').click()
  await expect(page.getByTestId('voice-picker')).toBeVisible()
  await expect(page.getByTestId('voice-about')).toContainText('SYSPIN')
  await expect(page.getByTestId('voices-missing')).toContainText('Server mode is unaffected')
  await page.getByTestId('tts-mode-server').click()
  await expect(page.getByLabel('Transport')).toBeVisible()
})
