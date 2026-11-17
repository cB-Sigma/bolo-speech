import { expect, test } from '@playwright/test'
import { MIRROR, openDemo, overflow } from './open'

test('the repetition guard is set like the server’s max_repeats and kept', async ({ page }) => {
  await openDemo(page, 'stt')
  const guard = page.getByTestId('repeat-guard')
  await expect(guard).toContainText('max_repeats')
  const select = guard.getByRole('combobox', { name: 'Repetition guard' })
  await expect(select).toContainText('4 repeats (default)')
  await select.click()
  await page.getByRole('option', { name: 'Off' }).click()
  await expect(select).toContainText('Off')
  await page.reload()
  await expect(page.getByTestId('repeat-guard').getByRole('combobox')).toContainText('Off')
  await page.setViewportSize({ width: 390, height: 900 })
  expect(await overflow(page)).toBeLessThanOrEqual(1)
})

test.describe('with the recogniser available', () => {
  test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a local model mirror to run')

  test('a sample goes through the guard and comes out unchanged when nothing repeats', async ({ page }) => {
    await openDemo(page, 'stt')
    await page.getByTestId('hero-sample').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 120_000 })
    await expect(page.getByTestId('transcript')).toContainText(/my fellow americans/i)
    await expect(page.getByTestId('repeats-removed')).toHaveCount(0)
  })
})
