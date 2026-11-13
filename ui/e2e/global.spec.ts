import { expect, test } from '@playwright/test'
import { open, expectNoA11yViolations } from './helpers'

test.use({ viewport: { width: 1440, height: 900 } })

test('command palette navigates and switches model with the keyboard only', async ({ page }) => {
  await open(page, '/stt')
  await page.keyboard.press('Control+k')
  const palette = page.getByRole('dialog', { name: 'Command palette' })
  await expect(palette).toBeVisible()
  await page.waitForTimeout(400) // let the open animation finish before measuring contrast
  await expectNoA11yViolations(page)
  await page.keyboard.type('latency')
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/#\/latency$/)
  await expect(palette).toBeHidden()

  await page.keyboard.press('Control+k')
  await page.keyboard.type('zipformer')
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/#\/stt$/)
  await expect(page.getByTestId('model-picker')).toContainText('zipformer-streaming-en')
})

test('keyboard: skip link, go-to shortcuts and focus are usable without a mouse', async ({ page }) => {
  await open(page, '/models')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.locator('#main')).toBeFocused()
  await page.keyboard.press('g')
  await page.keyboard.press('5')
  await expect(page).toHaveURL(/#\/benchmark$/)
  // Every focused control shows a visible focus indicator.
  await page.keyboard.press('Tab')
  await page.waitForTimeout(300) // focus rings fade in
  const outline = await page.evaluate(() => {
    const el = document.activeElement as HTMLElement
    const cs = getComputedStyle(el)
    return cs.outlineStyle !== 'none' || cs.boxShadow !== 'none'
  })
  expect(outline).toBe(true)
})

test('theme toggle switches and persists', async ({ page }) => {
  await open(page, '/connect', { theme: 'dark' })
  await expect(page.locator('html')).toHaveClass(/dark/)
  await page.getByRole('button', { name: /^Theme/ }).click()
  await page.getByRole('menuitemradio', { name: 'Light' }).click()
  await expect(page.locator('html')).not.toHaveClass(/dark/)
  await page.reload()
  await expect(page.locator('html')).not.toHaveClass(/dark/)
})

test('mobile navigation drawer', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await open(page, '/stt')
  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page.getByRole('dialog').getByRole('link', { name: 'Models' }).click()
  await expect(page).toHaveURL(/#\/models$/)
  await expect(page.getByRole('dialog')).toBeHidden()
  await expect(page.getByRole('heading', { level: 1, name: 'Models' })).toBeVisible()
})

test('asks for an API key when the server requires one, then connects', async ({ page }) => {
  await open(page, '/models', { mock: { auth: true } })
  const dialog = page.getByRole('dialog', { name: 'API key' })
  await expect(dialog).toBeVisible()
  await expectNoA11yViolations(page)
  await dialog.getByLabel('Key').fill('wrong-key')
  await dialog.getByRole('button', { name: 'Save key' }).click()
  await expect(page.getByRole('dialog', { name: 'API key' })).toContainText('rejected', { timeout: 10_000 })
  await page.getByRole('dialog', { name: 'API key' }).getByLabel('Key').fill('bolo-demo')
  await page.getByRole('dialog', { name: 'API key' }).getByRole('button', { name: 'Save key' }).click()
  await expect(page.getByTestId('connection')).toHaveAttribute('data-state', 'online', { timeout: 10_000 })
  await expect(page.getByTestId('model-card').first()).toBeVisible()
})

test('an unreachable server is reported with designed error states, not a blank page', async ({ page }) => {
  await open(page, '/models', { mock: { offline: true } })
  await expect(page.getByRole('alert').filter({ hasText: 'Server unreachable' })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('connection')).toHaveAttribute('data-state', 'offline')
  await expect(page.getByText('Could not load the catalog')).toBeVisible({ timeout: 15_000 })
  await expectNoA11yViolations(page)
  await page.goto('/#/latency')
  await expect(page.getByRole('heading', { level: 1, name: 'Latency' }).first()).toBeVisible()
  await page.goto('/#/connect')
  await expect(page.getByText('Server details unavailable')).toBeVisible({ timeout: 15_000 })
})

test('works when mounted under a sub-path (relative assets, hash routing)', async ({ page, baseURL }) => {
  // Proxy /console/* to the bundle root to imitate a reverse proxy mount.
  await page.route('**/console/**', async (route) => {
    const url = route.request().url().replace('/console/', '/')
    const res = await page.request.fetch(url)
    await route.fulfill({ response: res })
  })
  await page.goto(`${baseURL}/console/#/connect`)
  await expect(page.getByRole('heading', { level: 1, name: 'Connect' }).first()).toBeVisible()
  await expect(page.locator('pre').first()).toContainText(`${baseURL}/console/v1/audio/transcriptions`)
})
