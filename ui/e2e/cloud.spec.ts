import { expect, test } from '@playwright/test'
import { expectNoA11yViolations, expectNoHorizontalScroll, open, watchErrors, type Theme } from './helpers'

const SIZES = [
  { name: 'phone', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]
const THEMES: Theme[] = ['dark', 'light']

for (const size of SIZES) {
  for (const theme of THEMES) {
    test.describe(`cloud screens · ${size.name} · ${theme}`, () => {
      test.use({ viewport: { width: size.width, height: size.height }, colorScheme: theme })

      for (const route of [
        { path: '/providers', title: 'Providers' },
        { path: '/compare', title: 'Compare' },
      ]) {
        test(`${route.title} renders, fits the viewport and passes axe`, async ({ page }) => {
          const errors = watchErrors(page)
          await open(page, route.path, { theme })
          await expect(page.getByRole('heading', { level: 1, name: route.title }).first()).toBeVisible()
          await page.waitForTimeout(1200)
          await expect(page.locator('[aria-busy="true"]')).toHaveCount(0)
          await expectNoHorizontalScroll(page)
          await expectNoA11yViolations(page)
          expect(errors).toEqual([])
        })
      }
    })
  }
}

test.describe('Providers', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('add a provider with a write-only key, test it, edit it and remove it', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/providers')
    await expect(page.getByTestId('cloud-notice')).toContainText('Paid cloud API; audio leaves this server')
    const openai = page.getByTestId('provider-openai')
    await expect(openai).toContainText('from the configuration')
    await expect(openai.getByRole('button', { name: 'Edit' })).toHaveCount(0)
    await openai.getByRole('button', { name: 'Test' }).click()
    await expect(openai.getByTestId('provider-test-openai')).toContainText('The key was accepted')

    await page.getByRole('button', { name: 'Add provider' }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('combobox', { name: 'API' }).click()
    await page.getByRole('option', { name: 'Sarvam' }).click()
    await dialog.getByLabel('Name').fill('sarvam')
    await expect(dialog.getByTestId('provider-key')).toHaveAttribute('type', 'password')
    await dialog.getByTestId('provider-key').fill('sv-not-a-real-key')
    await dialog.getByRole('button', { name: 'Add a price' }).click()
    await dialog.getByLabel('Price 1: model').fill('saaras:v4')
    await dialog.getByLabel('Price 1: amount').fill('30')
    await expectNoA11yViolations(page)
    await dialog.getByRole('button', { name: 'Add provider' }).click()
    const sarvam = page.getByTestId('provider-sarvam')
    await expect(sarvam.getByTestId('provider-state-sarvam')).toContainText('Configured')
    await expect(page.locator('body')).not.toContainText('sv-not-a-real-key')

    // the stored key is never shown again: the field of the edit dialog is empty
    await sarvam.getByRole('button', { name: 'Edit' }).click()
    await expect(page.getByRole('dialog').getByTestId('provider-key')).toHaveValue('')
    await expect(page.getByRole('dialog').getByLabel('Price 1: model')).toHaveValue('saaras:v4')
    await page.getByRole('dialog').getByRole('button', { name: 'Cancel' }).click()

    // its cloud models are usable now, and marked
    await page.getByRole('link', { name: 'Models' }).first().click()
    await page.getByRole('combobox', { name: 'Runs' }).click()
    await page.getByRole('option', { name: 'Cloud' }).click()
    const card = page.getByTestId('model-card').filter({ hasText: 'sarvam-saaras-v4' })
    await expect(card.getByTestId('cloud-mark')).toBeVisible()
    await expect(card.getByText('Installed', { exact: true })).toBeVisible()
    await expect(page.getByTestId('model-card').filter({ hasText: 'moonshine-tiny-en' })).toHaveCount(0)
    await card.getByRole('button').first().click()
    await expect(page.getByRole('dialog').getByTestId('cloud-details')).toContainText('30 USD per minute of audio')
    await page.keyboard.press('Escape')

    await page.getByRole('link', { name: 'Providers' }).first().click()
    await page.getByTestId('provider-sarvam').getByRole('button', { name: 'Remove sarvam' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Remove' }).click()
    await expect(page.getByTestId('provider-sarvam')).toHaveCount(0)
    expect(errors).toEqual([])
  })
})

test.describe('Compare', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('the same sample through a local and a cloud model, side by side', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/compare')
    const models = page.getByTestId('compare-models')
    await expect(models.getByLabel('openai-gpt-4o-transcribe')).toBeChecked()
    await expect(page.getByTestId('compare-cloud-notice')).toContainText('openai')
    await page.getByTestId('compare-run').click()
    const cloud = page.getByTestId('compare-result-openai-gpt-4o-transcribe')
    await expect(cloud.getByTestId('compare-text')).toBeVisible({ timeout: 15_000 })
    await expect(cloud.getByTestId('cloud-mark')).toBeVisible()
    await expect(cloud).toContainText(/Estimated cost\s*0\.\d+ USD/)
    await expect(page.getByTestId('compare-results')).toContainText('none (runs here)')
    await expect(page.getByTestId('compare-results')).toContainText('Fastest')
    await expectNoA11yViolations(page)

    // text to speech: every chosen model answers with a player
    await page.getByRole('radio', { name: 'Text to speech' }).click()
    await expect(models.getByLabel('openai-gpt-4o-mini-tts')).toBeChecked()
    await page.getByTestId('compare-run').click()
    await expect(page.getByTestId('compare-result-openai-gpt-4o-mini-tts').locator('audio')).toBeVisible({
      timeout: 15_000,
    })
    expect(errors).toEqual([])
  })
})
