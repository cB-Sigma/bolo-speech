import { expect, test } from '@playwright/test'
import { expectNoA11yViolations, expectNoHorizontalScroll, open } from './helpers'

const KOKORO = { 'bolo.prefs': JSON.stringify({ ttsModel: 'kokoro-82m' }) }
// The mock's default: eSpeak NG is not installed, and the server may install it.
const MISSING = { mock: { espeak: false } }

test.describe('optional components (eSpeak NG)', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('Models: mark on voices that need it, the filter, and the components card', async ({ page }) => {
    await open(page, '/models', MISSING)
    const card = page.getByTestId('components-card')
    const row = card.getByTestId('component-espeak-ng')
    await expect(row).toContainText('eSpeak NG')
    await expect(row).toContainText('Not installed')
    await expect(row).toContainText('needed by 2 models')
    await expect(row.getByRole('link', { name: 'GPL-3.0-or-later' })).toHaveAttribute('href', /espeak-ng/)

    const kokoro = page.getByTestId('model-card').filter({ hasText: 'kokoro-82m' })
    await expect(kokoro).toContainText('Needs eSpeak NG')
    await expect(kokoro.getByRole('button', { name: 'Load' })).toBeDisabled()
    await expect(page.getByTestId('model-card').filter({ hasText: 'piper-en-us-lessac' })).not.toContainText(
      'Needs eSpeak NG',
    )

    await page.getByRole('switch', { name: 'No extra components' }).click()
    await expect(page.getByTestId('model-card').filter({ hasText: 'kokoro-82m' })).toHaveCount(0)
    await expect(page.getByTestId('model-card').filter({ hasText: 'piper-en-us-lessac' })).toBeVisible()
    await page.getByRole('button', { name: 'Clear' }).click()

    // Details drawer explains it too, and the table view carries the mark.
    await kokoro.getByRole('button').first().click()
    await expect(page.getByRole('dialog').getByTestId('component-notice')).toContainText('kokoro-82m needs eSpeak NG')
    await page.keyboard.press('Escape')
    await page.getByRole('radio', { name: 'Table' }).click()
    await expect(page.getByTestId('model-row').filter({ hasText: 'kokoro-82m' })).toContainText('Needs eSpeak NG')
    await expectNoA11yViolations(page)
  })

  for (const theme of ['dark', 'light'] as const) {
    test(`TTS: inline notice → confirm licence → progress → voice usable without reload · ${theme}`, async ({
      page,
    }) => {
      await open(page, '/tts', { ...MISSING, theme, storage: KOKORO })
      const notice = page.getByTestId('component-notice')
      await expect(notice).toContainText('kokoro-82m needs eSpeak NG')
      await expect(notice).toContainText('optional')
      await expect(notice).toContainText('not part of bolo')
      await expect(notice).toContainText('GPL-3.0-or-later')
      // It is a notice in the page, not a modal: the rest of the screen stays usable.
      await expect(page.getByRole('dialog')).toHaveCount(0)
      await expect(page.getByLabel('Text to speak')).toBeEditable()
      await expect(page.getByTestId('speak')).toBeDisabled()
      await expectNoA11yViolations(page)

      await notice.getByTestId('install-component').click()
      const dialog = page.getByTestId('install-dialog')
      await expect(dialog.getByRole('link', { name: /GPL-3\.0-or-later/ })).toHaveAttribute('href', /COPYING/)
      await expect(dialog).toContainText("A Python package is installed into the server's own environment")
      await expectNoA11yViolations(page)
      await dialog.getByTestId('install-confirm').click()
      await expect(dialog.getByRole('progressbar')).toBeVisible()
      await expect(page.getByText('eSpeak NG installed')).toBeVisible({ timeout: 15_000 })
      await expect(dialog).toBeHidden()

      // No reload: the notice is gone and the voice speaks.
      await expect(notice).toHaveCount(0)
      await expect(page.getByTestId('speak')).toBeEnabled()
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('ttfa')).toHaveText(/^\d+$/, { timeout: 10_000 })
      await page.getByTestId('tts-stop').click()
    })
  }

  test('TTS: when the server cannot install it, the reason and a copyable command are shown instead', async ({
    page,
  }) => {
    await open(page, '/tts', { mock: { espeak: false, noInstall: true }, storage: KOKORO })
    const notice = page.getByTestId('component-notice')
    await expect(notice).toContainText('The server cannot install it for you')
    await expect(notice).toContainText('component installs disabled')
    await expect(notice.getByTestId('install-component')).toHaveCount(0)
    await expect(notice.getByLabel('Manual install command')).toContainText('pip install')
    await notice.getByRole('button', { name: 'Copy' }).click()
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('pip install')
    await expectNoA11yViolations(page)
  })

  test('Connect: component status row, updated live when it is installed elsewhere', async ({ page }) => {
    await open(page, '/connect', MISSING)
    const row = page.getByTestId('component-espeak-ng')
    await expect(row).toContainText('Not installed')
    await row.getByTestId('install-espeak-ng').click()
    await page.getByTestId('install-confirm').click()
    await expect(row).toContainText('Installed', { timeout: 15_000 })
    await expect(row).toContainText('Python package')
    await expect(row.getByTestId('install-espeak-ng')).toHaveCount(0)
  })

  test('phone width: notices and the components card fit and pass axe', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    for (const [route, storage] of [
      ['/tts', KOKORO],
      ['/models', undefined],
      ['/connect', undefined],
    ] as const) {
      await open(page, route, { ...MISSING, storage })
      await page.goto(`/#${route}`)
      await expect(page.getByText('eSpeak NG').first()).toBeVisible()
      await expectNoHorizontalScroll(page)
      await expectNoA11yViolations(page)
    }
  })
})

test.describe('language packs', () => {
  test('Connect: one summary row, a paged list behind View, at desktop and phone width', async ({ page }) => {
    for (const viewport of [
      { width: 1400, height: 1000 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport)
      await open(page, '/connect')
      const packs = page.getByTestId('language-packs')
      await expect(packs.getByTestId('language-packs-summary')).toContainText('2 of 35 installed')
      await expect(page.getByTestId('component-uccharan-de')).toHaveCount(0)
      await packs.getByRole('button', { name: 'View' }).click()
      await expect(page.getByTestId('component-uccharan-de')).toContainText('German')
      await page.getByTestId('packs-filter-all').click()
      await expect(packs.getByTestId('language-pack-list').getByRole('listitem')).toHaveCount(8)
      await expect(packs.getByTestId('packs-range')).toHaveText('1–8 of 35')
      await expectNoHorizontalScroll(page)
      await expectNoA11yViolations(page)
      await packs.getByRole('button', { name: 'Hide' }).click()
    }
  })
})
