import AxeBuilder from '@axe-core/playwright'
import { expect, type Page } from '@playwright/test'

export const ROUTES = [
  { path: '/stt', title: 'Speech to text' },
  { path: '/tts', title: 'Text to speech' },
  { path: '/models', title: 'Models' },
  { path: '/latency', title: 'Latency' },
  { path: '/benchmark', title: 'Benchmark' },
  { path: '/connect', title: 'Connect' },
] as const

export type Theme = 'dark' | 'light'

/** Open a screen with a fixed theme and optional mock-server flags. */
export async function open(
  page: Page,
  route: string,
  opts: { theme?: Theme; mock?: Record<string, boolean>; storage?: Record<string, string> } = {},
) {
  await page.addInitScript(
    ({ theme, mock, storage }) => {
      if (sessionStorage.getItem('e2e.init')) return // only seed once per tab, so reloads keep app state
      sessionStorage.setItem('e2e.init', '1')
      localStorage.setItem('bolo.theme', theme)
      if (mock) localStorage.setItem('bolo.mock', JSON.stringify(mock))
      for (const [k, v] of Object.entries(storage ?? {})) localStorage.setItem(k, v)
    },
    // Most tests are not about optional components, so the mock starts with eSpeak NG present unless told otherwise.
    { theme: opts.theme ?? 'dark', mock: opts.mock ?? { espeak: true }, storage: opts.storage },
  )
  await page.goto(`/#${route}`)
  await expect(page.locator('#main')).toBeVisible()
}

/** Console errors and uncaught exceptions seen on the page. */
export function watchErrors(page: Page): string[] {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('pageerror', (e) => errors.push(e.message))
  return errors
}

export async function expectNoA11yViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'])
    .analyze()
  const summary = results.violations.map(
    (v) =>
      `${v.id} (${v.impact}): ${v.help}\n    ${v.nodes
        .slice(0, 3)
        .map((n) => n.target.join(' '))
        .join('\n    ')}`,
  )
  expect(summary, summary.join('\n')).toEqual([])
}

export async function expectNoHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow, 'page should not scroll sideways').toBeLessThanOrEqual(1)
}
