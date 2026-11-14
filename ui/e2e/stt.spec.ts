import path from 'node:path'
import { expect, test } from '@playwright/test'
import { open, watchErrors } from './helpers'

test.use({ viewport: { width: 1440, height: 900 } })

test('microphone: fake device → AudioWorklet → websocket → live transcript and latency', async ({ page }) => {
  const errors = watchErrors(page)
  await open(page, '/stt')
  await expect(page.getByTestId('model-picker')).toContainText('moonshine-tiny-en')
  await page.getByTestId('mic').click()
  await expect(page.getByTestId('stop')).toBeVisible()
  // The fake microphone plays real speech, so the mock recogniser's VAD fires on it.
  await expect(page.getByTestId('partial-text')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('latency-timeline').getByRole('img')).toHaveAccessibleName(
    /final \d+ ms after speech ended/,
  )
  await expect(page.getByTestId('budget-bar')).toHaveAttribute('data-state', /good|warning|bad/)
  await expect(page.getByTestId('rtt')).toContainText(/network \d+ ms/)
  await page.getByTestId('stop').click()
  await expect(page.getByTestId('mic')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Replay' })).toBeVisible()
  // Session statistics filled in.
  await expect(page.getByTestId('readout').filter({ hasText: 'First partial p50' })).toContainText(/\d+\s*ms/)
  expect(errors).toEqual([])
})

test('built-in sample streams at real-time pace with word timestamps on hover', async ({ page }) => {
  await open(page, '/stt')
  await page.getByTestId('sample-en-fellow-americans').click()
  await expect(page.getByTestId('final-text').first()).toContainText(/fellow Americans/i, { timeout: 15_000 })
  await expect(page.getByTestId('mic')).toBeVisible({ timeout: 10_000 })
  await page.getByTestId('final-text').first().getByText('fellow').hover()
  await expect(page.getByRole('tooltip')).toContainText(/\d\.\d\d–\d\.\d\d s/)
  await page.getByRole('button', { name: 'Copy' }).click()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(/fellow Americans/i)
})

test('file upload and the phone-line simulation', async ({ page }) => {
  await open(page, '/stt')
  await page.getByRole('switch', { name: /Phone line/ }).click()
  await page
    .getByLabel('Audio file', { exact: true })
    .setInputFiles(path.resolve(import.meta.dirname, '../src/mock/assets/en-fellow-americans.wav'))
  await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('mic')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByRole('button', { name: 'Hear it as a phone line' })).toBeVisible()
  await expect(page.getByText(/Last take: en-fellow-americans\.wav/)).toBeVisible()
})

test('changing the latency target updates the budget', async ({ page }) => {
  await open(page, '/stt')
  await page.getByTestId('sample-en-fellow-americans').click()
  await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('mic')).toBeVisible({ timeout: 10_000 })
  await page.getByRole('button', { name: /Target/ }).click()
  await page.getByLabel('Latency target (ms)').fill('5000')
  await expect(page.getByTestId('budget-bar')).toHaveAttribute('data-state', 'good')
  await page.getByLabel('Latency target (ms)').fill('50')
  await expect(page.getByTestId('budget-bar')).toHaveAttribute('data-state', 'bad')
})

test('a model that is not installed is explained, not silently broken', async ({ page }) => {
  await open(page, '/stt', { storage: { 'bolo.prefs': JSON.stringify({ sttModel: 'whisper-small' }) } })
  await expect(page.getByText('whisper-small is not installed')).toBeVisible()
  await expect(page.getByTestId('mic')).toBeDisabled()
})
