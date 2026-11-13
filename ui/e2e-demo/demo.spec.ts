import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

const MIRROR = Boolean(process.env.BOLO_E2E_MODEL_DIR)
const CDN = 'http://127.0.0.1:4175'
const ROUTES = ['stt', 'tts', 'models', 'latency', 'benchmark', 'connect']

/** Opens the demo and waits out the one-off reload that makes the page cross-origin isolated. */
async function open(page: Page, route = '', opts: { theme?: 'dark' | 'light' } = {}) {
  await page.addInitScript(
    ([theme, cdn]) => {
      localStorage.setItem('bolo.theme', theme as string)
      // Stand-in for huggingface.co and jsDelivr (see playwright.demo.config.ts).
      localStorage.setItem('bolo.browserStt', JSON.stringify({ modelHost: cdn, ortWasmBase: `${cdn}/ort/` }))
    },
    [opts.theme ?? 'dark', CDN],
  )
  await page.goto(`./#/${route}`)
  await expect(page.getByTestId('demo-banner')).toBeVisible()
  await expect.poll(() => page.evaluate(() => crossOriginIsolated), { timeout: 20_000 }).toBe(true)
  await expect(page.getByTestId('demo-banner')).toBeVisible()
}

async function axe(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'])
    .analyze()
  expect(
    results.violations.map(
      (v) =>
        `${v.id}: ${v.nodes
          .map((n) => n.target.join(' '))
          .slice(0, 3)
          .join(' | ')}`,
    ),
  ).toEqual([])
}

test('a Pages-like host sends no isolation headers; the service worker supplies them after one reload', async ({
  page,
}) => {
  const first = await page.request.get('./')
  expect(first.headers()['cross-origin-opener-policy']).toBeUndefined()
  expect(first.headers()['cross-origin-embedder-policy']).toBeUndefined()

  const started = Date.now()
  let navigations = 0
  page.on('framenavigated', (f) => f === page.mainFrame() && navigations++)
  await page.goto('./')
  await expect(page.getByTestId('demo-hero')).toBeVisible()
  const state = await page.evaluate(() => ({
    isolated: crossOriginIsolated,
    controlled: Boolean(navigator.serviceWorker.controller),
    sab: typeof SharedArrayBuffer !== 'undefined',
  }))
  console.log(
    `[demo] first visit usable after ${Date.now() - started} ms · ${JSON.stringify(state)} · navigations ${navigations}`,
  )
  expect(state).toEqual({ isolated: true, controlled: true, sab: true })
  expect(page.url()).toMatch(/\/bolo\/#\/stt$/)
  await expect(page.getByTestId('demo-banner')).toContainText('Everything here runs in your browser')
  await expect(page.getByTestId('connection')).toContainText('In browser')
  await expect(page).toHaveTitle(/Speech to text · bolo/)
  // Link previews: absolute image URL and a description.
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute(
    'content',
    'https://cb-sigma.github.io/bolo/og.png',
  )
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute('content', 'summary_large_image')
})

test('without a service worker the demo still loads, single-threaded', async ({ browser }) => {
  const context = await browser.newContext({ serviceWorkers: 'block' })
  const page = await context.newPage()
  await page.goto('./')
  await expect(page.getByTestId('demo-hero')).toBeVisible()
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(false)
  await page.goto('./#/connect')
  await expect(page.getByTestId('support-list')).toContainText('not cross-origin isolated')
  await context.close()
})

for (const theme of ['dark', 'light'] as const) {
  for (const width of [390, 1440]) {
    test(`every screen renders, fits and passes axe · ${theme} · ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 })
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(e.message))
      await open(page, 'stt', { theme })
      for (const route of ROUTES) {
        await page.goto(`./#/${route}`)
        await expect(page.locator('h1:visible').first()).toBeVisible()
        await page.waitForTimeout(700)
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
        )
        expect(overflow, `${route} should not scroll sideways`).toBeLessThanOrEqual(1)
        await axe(page)
      }
      expect(errors).toEqual([])
    })
  }
}

test('server-only features are disabled with an explanation and the install command', async ({ page }) => {
  await open(page, 'stt')
  await expect(page.getByRole('radio', { name: 'Server' })).toBeDisabled()
  await expect(page.getByRole('radio', { name: 'Compare' })).toBeDisabled()
  await expect(page.getByRole('radio', { name: 'In browser' })).toBeChecked()
  await expect(page.getByRole('switch', { name: /Phone line/ })).toBeDisabled()

  await page.goto('./#/tts')
  await expect(page.getByTestId('needs-server')).toContainText('Text to speech needs the server')
  await expect(page.getByLabel('Install command')).toContainText('pipx install')

  await page.goto('./#/models')
  await expect(page.getByRole('heading', { name: /Runs in your browser/ })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Speech to text.*available when you run the server/ })).toBeVisible()
  await page.getByLabel('Search models').fill('whisper-tiny')
  const serverCard = page.getByTestId('model-card').filter({ hasText: 'whisper-tiny' }).first()
  await expect(serverCard).toContainText('Needs the server')
  await expect(serverCard.getByRole('button', { name: /Pull/ })).toHaveCount(0)
  await page.getByLabel('Search models').fill('')
  const browserCard = page.getByTestId('model-card').filter({ hasText: 'onnxruntime-web' })
  await expect(browserCard).toContainText('MIT')
  await expect(browserCard.getByTestId('pull-moonshine-tiny')).toContainText('Download')

  // Optional components are listed read-only.
  const comp = page.getByTestId('components-card').getByTestId('component-espeak-ng')
  await expect(comp).toContainText('GPL-3.0-or-later')
  await expect(comp.getByRole('button')).toHaveCount(0)
  await expect(page.getByTestId('components-card')).toContainText('installing components is a server feature')

  await page.goto('./#/connect')
  await expect(page.getByTestId('needs-server')).toContainText('no API behind it')
  await expect(page.getByTestId('component-espeak-ng')).toContainText('Not installed')
  await expect(page.locator('pre').nth(1)).toContainText('http://127.0.0.1:8000/v1/audio/transcriptions')

  await page.goto('./#/latency')
  await expect(page.getByText('No requests in this window').first()).toBeVisible()
  await expect(page.getByText('a web page cannot measure its own')).toBeVisible()
  await page.goto('./#/benchmark')
  await expect(page.getByTestId('bench-demo-note')).toContainText('runs in your browser')
})

test('a shared link shows the sender’s numbers, labelled as theirs, and ignores junk', async ({ page }) => {
  await open(page, 'stt')
  const token = await page.evaluate(() =>
    btoa(JSON.stringify([1, 'Moonshine Tiny', 'WASM, 4 threads', 4, 410, 830, 990, 0.41])).replace(/=+$/, ''),
  )
  await page.goto(`./#/stt?r=${token}`)
  const card = page.getByTestId('shared-result')
  await expect(card).toContainText('someone else')
  await expect(card).toContainText('830')
  await expect(card).toContainText('WASM, 4 threads')
  await card.getByRole('button', { name: 'Dismiss' }).click()
  await expect(card).toBeHidden()
  await page.goto('./#/stt?r=%3Cscript%3E')
  await expect(page.getByTestId('demo-hero')).toBeVisible()
  await expect(page.getByTestId('shared-result')).toHaveCount(0)
})

test.describe('with the model available', () => {
  test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a local model mirror to run')

  test('one click: sample → real transcription with threads → real traces → share → benchmark → offline', async ({
    page,
    context,
  }) => {
    const errors: string[] = []
    page.on('pageerror', (e) => errors.push(e.message))
    await open(page, 'stt')

    // First run: download, compile, transcribe.
    const t0 = Date.now()
    await page.getByTestId('hero-sample').click()
    const engine = page.getByTestId('browser-engine')
    await expect(engine.getByTestId('engine-status')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 })
    const readyMs = Date.now() - t0
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 60_000 })
    const transcript = (await page.getByTestId('transcript').innerText()).toLowerCase()
    expect(transcript).toContain('my fellow americans')
    expect(transcript).toContain('what you can do for your country')
    await expect(engine).toContainText(/WebAssembly · [2-9] threads/)
    const p50 = await page.getByTestId('readout').filter({ hasText: 'Final p50' }).innerText()
    const rtf = await page.getByTestId('readout').filter({ hasText: 'Real-time factor' }).innerText()
    console.log(
      `[demo] model ready ${readyMs} ms · ${(await engine.innerText()).match(/WebAssembly · \d threads?/)?.[0]} · ${p50.replace(/\s+/g, ' ')} · ${rtf.replace(/\s+/g, ' ')}`,
    )

    // Share link carries the summary.
    await page.getByTestId('share').click()
    const link = await page.evaluate(() => navigator.clipboard.readText())
    expect(link).toMatch(/\/bolo\/#\/stt\?r=[A-Za-z0-9_-]+$/)
    const other = await context.newPage()
    await other.goto(link)
    await expect(other.getByTestId('shared-result')).toContainText('Moonshine Tiny')
    await other.close()

    // The dashboard shows what just ran, and nothing else.
    await page.goto('./#/latency')
    const rows = page.getByTestId('waterfall').getByRole('button')
    await expect(rows.first()).toBeVisible()
    const count = await rows.count()
    expect(count).toBeGreaterThanOrEqual(4) // model load + at least three utterances
    expect(count).toBeLessThanOrEqual(8)
    await expect(page.getByTestId('waterfall')).toContainText('moonshine-tiny')
    await rows.last().click()
    await expect(page.getByTestId('request-drawer')).toContainText('Model load')
    await page.keyboard.press('Escape')
    await rows.first().click()
    const drawer = page.getByTestId('request-drawer')
    await expect(drawer).toContainText('Endpointing')
    await expect(drawer).toContainText('Decode')
    await expect(drawer).toContainText('browser')
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('stat-tile').filter({ hasText: 'First partial p50' })).toContainText(/\d+\s*ms/)

    // Models: the browser model is now loaded.
    await page.goto('./#/models')
    await expect(page.getByTestId('model-card').filter({ hasText: 'onnxruntime-web' })).toContainText('Loaded')

    // A real in-page sweep.
    await page.goto('./#/benchmark')
    await page.getByLabel('Concurrency levels').fill('1, 2')
    await page.locator('#bench-duration').fill('2')
    await page.getByTestId('bench-run').click()
    await expect(page.getByText('Benchmark finished')).toBeVisible({ timeout: 120_000 })
    const tableRows = page.getByTestId('bench-table').locator('tbody tr')
    await expect(tableRows).toHaveCount(2)
    console.log(`[demo] bench: ${(await tableRows.allInnerTexts()).map((r) => r.replace(/\s+/g, ' ')).join(' || ')}`)

    // Offline: the shell, the runtime and the model all come from caches.
    await context.setOffline(true)
    await page.reload()
    await expect(page.getByTestId('demo-banner')).toBeVisible()
    expect(await page.evaluate(() => crossOriginIsolated)).toBe(true)
    await page.goto('./#/stt')
    const t1 = Date.now()
    await page.getByTestId('sample-en-fellow-americans').click()
    await expect(page.getByTestId('final-text').first()).toContainText(/fellow americans/i, { timeout: 90_000 })
    console.log(`[demo] offline: transcribed from cache ${Date.now() - t1} ms after the click`)
    await context.setOffline(false)
    expect(errors).toEqual([])
  })

  test('microphone path in the demo (fake device)', async ({ page }) => {
    await open(page, 'stt')
    await page.getByTestId('hero-mic').click()
    await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 150_000 })
    await page.getByTestId('stop').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 60_000 })
    await expect(page.getByTestId('latency-timeline').getByRole('img')).toHaveAccessibleName(
      /final \d[\d,]* ms after speech ended/,
    )
  })
})
