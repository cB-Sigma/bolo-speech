import path from 'node:path'
import { expect, test } from '@playwright/test'
import { expectNoA11yViolations, expectNoHorizontalScroll, open } from './helpers'

const fake = { 'bolo.mock.browserStt': JSON.stringify({ loadMs: 400 }) }

test.describe('Indian languages', () => {
  for (const width of [1440, 390]) {
    test(`in-browser: choosing Hindi offers a model that can do it, with the cost · ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 })
      await open(page, '/stt', { storage: fake })
      await page.getByTestId('mode-browser').click()
      const picker = page.getByTestId('browser-model-picker')
      await expect(picker).toContainText('Moonshine Tiny')
      const language = page.getByTestId('language-select')
      await expect(language).toContainText('English')

      // The picker lists every in-browser model with coverage, size and licence (the list grows; these stay).
      await picker.click()
      const models = page.getByRole('option')
      await expect(models.first()).toHaveText(/Moonshine Tiny.*English · 32\.1 MB.*MIT/)
      for (const name of [
        /^Whisper Tiny Hindi.*Hindi · 124 MB.*Apache-2\.0/,
        /^Whisper Tiny99 languages · 43\.6 MB.*Apache-2\.0/,
        /^Whisper Base99 languages · 79\.7 MB.*Apache-2\.0/,
        /^Moonshine Tiny Arabic.*Moonshine Community License/,
      ]) {
        await expect(models.filter({ hasText: name })).toHaveCount(1)
      }
      await page.keyboard.press('Escape')

      // Indian languages come first, under their own names.
      await language.click()
      const options = page.getByRole('option')
      await expect(options.nth(0)).toHaveText('हिन्दी · Hindi')
      await expect(options.nth(1)).toHaveText('বাংলা · Bengali')
      for (const name of ['தமிழ் · Tamil', 'తెలుగు · Telugu', 'मराठी · Marathi', 'ગુજરાતી · Gujarati', 'ಕನ್ನಡ · Kannada']) {
        await expect(page.getByRole('option', { name })).toBeVisible()
      }
      for (const name of [
        'മലയാളം · Malayalam',
        'ਪੰਜਾਬੀ · Punjabi',
        'اردو · Urdu',
        'नेपाली · Nepali',
        'অসমীয়া · Assamese',
      ]) {
        await expect(page.getByRole('option', { name })).toHaveCount(1)
      }
      await page.getByRole('option', { name: 'हिन्दी · Hindi' }).click()

      // Moonshine cannot do Hindi: the page asks before switching, and says what it costs.
      const dialog = page.getByTestId('switch-model-dialog')
      await expect(dialog.getByRole('heading')).toHaveText('Hindi needs a different model')
      await expect(dialog).toContainText('Moonshine Tiny only understands English')
      const choices = dialog.locator('button[data-testid^="switch-to-"]')
      // The suggestion first, then the smallest that can; every model that speaks Hindi is offered.
      await expect(choices.nth(0)).toHaveText(/Whisper Tiny Hindi.*Suggested.*124 MB download.*Apache-2\.0/)
      await expect(choices.nth(1)).toHaveText(/Whisper Tiny.*43\.6 MB download/)
      await expect(choices.nth(2)).toHaveText(/Whisper Base.*79\.7 MB download/)
      await expect(choices.filter({ hasText: /IndicWav2Vec Hindi \(AI4Bharat\)/ })).toHaveCount(1)
      await expectNoA11yViolations(page)
      await dialog.getByTestId('switch-to-whisper-tiny-hindi').click()
      await expect(picker).toContainText('Whisper Tiny Hindi')
      await expect(language).toContainText('हिन्दी · Hindi')

      // Honest about quality, with the way to better models.
      const note = page.getByTestId('whisper-note')
      await expect(note).toContainText('Expect mistakes')
      await expect(note).toContainText('word error rate 40%')
      await expect(note).toContainText('bolo server runs much better models for Indian languages')
      await expect(note).toContainText('uv tool install')

      // The transcript comes out in Devanagari, set in a font the page ships (no empty boxes).
      await page.getByTestId('model-chip').first().click()
      await expect(page.getByTestId('model-chip').first()).toContainText('Ready', { timeout: 15_000 })
      await page
        .locator('input[type=file]')
        .setInputFiles(path.resolve(import.meta.dirname, '../src/mock/assets/en-fellow-americans.wav'))
      await expect(page.getByTestId('final-text').first()).toContainText('नमस्ते', { timeout: 20_000 })
      await expect(page.getByTestId('mic')).toBeVisible({ timeout: 15_000 })
      await expect
        .poll(() => page.evaluate(() => document.fonts.check('16px "Noto Sans Devanagari Variable"', 'नमस्ते')))
        .toBe(true)
      const fontUsed = await page.evaluate(async () => {
        await document.fonts.ready
        return [...document.fonts].some((f) => f.family.includes('Noto Sans Devanagari') && f.status === 'loaded')
      })
      expect(fontUsed).toBe(true)
      await expectNoA11yViolations(page)
      await expectNoHorizontalScroll(page)
    })
  }

  test('in-browser: the multilingual models say plainly that they are not usable for Hindi', async ({ page }) => {
    await open(page, '/stt', {
      storage: {
        ...fake,
        'bolo.prefs': JSON.stringify({
          browserModel: 'whisper-base-q8',
          browserModelPicked: true,
          browserLanguage: 'hi',
        }),
      },
    })
    await page.getByTestId('mode-browser').click()
    await expect(page.getByTestId('browser-model-picker')).toContainText('Whisper Base')
    await expect(page.getByTestId('whisper-note')).toContainText('Not usable for Hindi')
    await expect(page.getByTestId('whisper-note')).toContainText('Choose Whisper Tiny Hindi instead')
    // Another Indian language: not measured, said so.
    await page.getByTestId('language-select').click()
    await page.getByRole('option', { name: 'தமிழ் · Tamil' }).click()
    await expect(page.getByTestId('whisper-note')).toContainText('Expect a rough guess in Tamil')
    // Auto-detect is offered for a multilingual model.
    await page.getByTestId('language-select').click()
    await expect(page.getByRole('option', { name: 'Auto-detect' })).toBeVisible()
  })

  test('server: choosing a language narrows the models and switches to one that supports it', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await open(page, '/stt')
    const picker = page.getByTestId('model-picker')
    await expect(picker).toContainText('moonshine-tiny-en')
    const language = page.getByTestId('language-select')
    await language.click()
    await expect(page.getByRole('option').nth(0)).toHaveText('Auto-detect')
    await expect(page.getByRole('option').nth(1)).toHaveText('हिन्दी · Hindi')
    await page.getByRole('option', { name: 'हिन्दी · Hindi' }).click()
    // moonshine-tiny-en is English only.
    await expect(picker).toContainText('indic-conformer-hi')
    await expect(page.getByText('Switched to indic-conformer-hi')).toBeVisible()
    await picker.click()
    const ids = await page.getByRole('option').allInnerTexts()
    expect(ids.some((t) => t.includes('indic-conformer-hi'))).toBe(true)
    expect(ids.some((t) => t.includes('whisper-base'))).toBe(true) // multilingual
    expect(ids.some((t) => t.includes('moonshine-tiny-en'))).toBe(false)
    await expect(page.getByTestId('picker-language-note')).toContainText('Showing models that support Hindi')
    await page.keyboard.press('Escape')
    // Back to auto: everything is offered again.
    await language.click()
    await page.getByRole('option', { name: 'Auto-detect' }).click()
    await picker.click()
    await expect(page.getByRole('option', { name: /moonshine-tiny-en/ })).toBeVisible()
    await expect(page.getByTestId('picker-language-note')).toHaveCount(0)
  })

  test('the model catalog can be filtered by Hindi', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await open(page, '/models')
    await page.getByRole('combobox', { name: 'Language' }).click()
    await page.getByRole('option', { name: 'हिन्दी · Hindi' }).click()
    await expect(page.getByTestId('model-card').filter({ hasText: 'indic-conformer-hi' })).toBeVisible()
    await expect(page.getByTestId('model-card').filter({ hasText: 'moonshine-tiny-en' })).toHaveCount(0)
  })
})
