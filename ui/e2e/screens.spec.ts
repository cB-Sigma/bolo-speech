import { expect, test } from '@playwright/test'
import { expectNoA11yViolations, expectNoHorizontalScroll, open, ROUTES, watchErrors, type Theme } from './helpers'

const SIZES = [
  { name: 'phone', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]
const THEMES: Theme[] = ['dark', 'light']

for (const size of SIZES) {
  for (const theme of THEMES) {
    test.describe(`${size.name} · ${theme}`, () => {
      test.use({ viewport: { width: size.width, height: size.height }, colorScheme: theme })

      for (const route of ROUTES) {
        test(`${route.title} renders, fits the viewport and passes axe`, async ({ page }) => {
          const errors = watchErrors(page)
          await open(page, route.path, { theme })
          await expect(page.getByRole('heading', { level: 1, name: route.title }).first()).toBeVisible()
          await expect(page.locator('html')).toHaveClass(theme === 'dark' ? /dark/ : /^(?!.*dark)/)
          await expect(page.getByTestId('connection')).toHaveAttribute('data-state', 'online')
          // Let queries settle and live data arrive before judging the page.
          await page.waitForTimeout(1500)
          await expect(page.locator('[aria-busy="true"]')).toHaveCount(0)
          await expectNoHorizontalScroll(page)
          await expectNoA11yViolations(page)
          expect(errors).toEqual([])
        })
      }
    })
  }
}
