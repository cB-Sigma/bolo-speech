import { expect, type Page } from '@playwright/test'

export const MIRROR = Boolean(process.env.BOLO_E2E_MODEL_DIR)
export const CDN = 'http://127.0.0.1:4175'

/** Opens the demo with the local stand-in for the model and runtime CDNs, after the isolation reload. */
export async function openDemo(page: Page, route = '', opts: { theme?: 'dark' | 'light' } = {}): Promise<void> {
  await page.addInitScript(
    ([theme, cdn]) => {
      localStorage.setItem('bolo.theme', theme as string)
      localStorage.setItem(
        'bolo.browserStt',
        JSON.stringify({
          modelHost: cdn,
          ortWasmBase: `${cdn}/ort/`,
          npmBase: `${cdn}/npm/`,
          // The demo serves its copy of GTCRN next to the page; the mirror stands in for it.
          modelsBase: `${cdn}/site-models/`,
        }),
      )
      localStorage.setItem('bolo.browserTts', JSON.stringify({ hubHost: cdn }))
    },
    [opts.theme ?? 'dark', CDN],
  )
  await page.goto(`./#/${route}`)
  await expect(page.getByTestId('demo-banner')).toBeVisible()
  await expect.poll(() => page.evaluate(() => crossOriginIsolated), { timeout: 20_000 }).toBe(true)
  await expect(page.getByTestId('demo-banner')).toBeVisible()
}

/** The page scrolls sideways by this many pixels (0 when it fits). */
export function overflow(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
}
