import { expect, test } from '@playwright/test'
import { expectNoA11yViolations, open, watchErrors } from './helpers'

test.use({ viewport: { width: 1440, height: 900 } })

test.describe('Loaded models', () => {
  test('lists what is in memory with placement and a countdown; keep loaded and unload', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/models')
    const panel = page.getByTestId('loaded-models')
    await expect(panel.getByRole('heading', { name: 'Loaded models' })).toBeVisible()
    await expect(panel.getByTestId('machine-totals')).toContainText('CPU')
    await expect(panel.getByTestId('machine-totals')).toContainText('RAM')
    const rows = panel.getByTestId('loaded-model')
    await expect(rows.first()).toBeVisible()
    // the mock machine has no GPU: everything runs on the CPU
    await expect(rows.first().getByTestId('processor')).toHaveText('100% CPU')

    const row = rows.filter({ hasNot: page.getByText('Kept loaded') }).first()
    const id = await row.getAttribute('data-model')
    expect(id).toBeTruthy()
    await expect(row.getByTestId('countdown')).toHaveText(/^\d+:\d\d$/)
    await row.getByRole('button', { name: `Keep loaded: ${id}` }).click()
    await expect(page.getByText(`${id} stays loaded`)).toBeVisible()
    const same = panel.locator(`[data-model="${id}"]`)
    await expect(same.getByTestId('countdown')).toHaveText('Kept loaded')
    await same.getByRole('button', { name: `Unload ${id}` }).click()
    await expect(panel.locator(`[data-model="${id}"]`)).toHaveCount(0)
    await expectNoA11yViolations(page)
    expect(errors).toEqual([])
  })
})
