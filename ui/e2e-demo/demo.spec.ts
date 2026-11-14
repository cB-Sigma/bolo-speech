import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

const MIRROR = Boolean(process.env.BOLO_E2E_MODEL_DIR)
const VOICES = Boolean(process.env.BOLO_E2E_VOICE_DIR)
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

  await page.goto('./#/models')
  await expect(page.getByRole('heading', { name: /Runs in your browser/ })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Speech to text.*available when you run the server/ })).toBeVisible()
  await page.getByLabel('Search models').fill('faster-whisper-tiny')
  const serverCard = page.getByTestId('model-card').filter({ hasText: 'faster-whisper-tiny' }).first()
  await expect(serverCard).toContainText('Needs the server')
  await expect(serverCard.getByRole('button', { name: /Pull/ })).toHaveCount(0)
  await page.getByLabel('Search models').fill('')
  // Four models run in the page; each says what it is and what it costs.
  await expect(page.getByTestId('model-card').filter({ hasText: 'onnxruntime-web' })).toHaveCount(4)
  const browserCard = page.getByTestId('model-card').filter({ has: page.getByTestId('pull-moonshine-tiny') })
  await expect(browserCard).toContainText('MIT')
  await expect(
    page.getByTestId('model-card').filter({ has: page.getByTestId('pull-whisper-tiny-hindi') }),
  ).toContainText('Apache-2.0')
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
    await expect(page.getByTestId('model-card').filter({ hasText: 'onnxruntime-web' }).first()).toContainText('Loaded')

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

test.describe('first visit: microphone and model status', () => {
  test('the hero says what the microphone and the model will do, and offers Hindi', async ({ page }) => {
    await open(page, 'stt')
    const hero = page.getByTestId('demo-hero')
    // The test browser has the permission granted.
    await expect(hero.getByTestId('mic-status')).toHaveAttribute('data-permission', 'granted')
    await expect(hero.getByTestId('mic-status')).toContainText('allowed for this site')
    await expect(hero.getByTestId('model-chip')).toContainText('Moonshine Tiny')
    await expect(hero).toContainText('32.1 MB')
    await expect(page.getByTestId('sample-hi-short')).toBeVisible()
    await hero.getByTestId('hero-hindi').click()
    const dialog = page.getByTestId('switch-model-dialog')
    await expect(dialog.getByRole('heading')).toHaveText('Hindi needs a different model')
    await expect(dialog.getByTestId('switch-to-whisper-tiny-hindi')).toContainText('Suggested')
    await expect(dialog.getByTestId('switch-to-whisper-tiny-hindi')).toContainText('124 MB download')
    await axe(page)
    await dialog.getByRole('button', { name: 'Keep Moonshine Tiny' }).click()
    await expect(page.locator('meta[name="viewport"]')).toHaveAttribute('content', /viewport-fit=cover/)
  })

  test('Moonshine is the recogniser until the visitor picks another one themselves', async ({ page }) => {
    // An earlier version stored the Hindi model after a speech-synthesis round trip; nobody chose it.
    await page.addInitScript(() => {
      if (!localStorage.getItem('bolo.prefs'))
        localStorage.setItem('bolo.prefs', JSON.stringify({ browserModel: 'whisper-tiny-hindi' }))
    })
    await open(page, 'stt')
    await expect(page.getByTestId('demo-hero').getByTestId('model-chip')).toContainText('Moonshine Tiny')
    // Choosing Hindi still offers the Hindi model, and a choice made here is kept.
    await page.getByTestId('demo-hero').getByTestId('hero-hindi').click()
    await page.getByTestId('switch-model-dialog').getByTestId('switch-to-whisper-tiny-hindi').click()
    await expect
      .poll(() =>
        page.evaluate(() => JSON.parse(localStorage.getItem('bolo.prefs') ?? '{}') as Record<string, unknown>),
      )
      .toMatchObject({ browserModel: 'whisper-tiny-hindi', browserModelPicked: true })
    await page.reload()
    await expect(page.getByTestId('demo-banner')).toBeVisible()
    await expect(page.getByTestId('browser-model-picker')).toContainText('Whisper Tiny Hindi')
  })

  test('a blocked microphone is flagged before the tap and explained after it, at phone width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = () =>
        new Promise((_r, reject) =>
          setTimeout(() => reject(new DOMException('Permission denied', 'NotAllowedError')), 5),
        )
      const query = navigator.permissions.query.bind(navigator.permissions)
      navigator.permissions.query = (d: PermissionDescriptor) =>
        (d.name as string) === 'microphone'
          ? Promise.resolve({ state: 'denied', onchange: null } as unknown as PermissionStatus)
          : query(d)
    })
    await open(page, 'stt')
    const status = page.getByTestId('mic-status')
    await expect(status).toContainText('blocked for this site')
    // The explanation is available before trying.
    await status.getByRole('button', { name: 'How to allow it' }).click()
    await expect(page.getByTestId('mic-help').getByRole('heading')).toHaveText(
      'The microphone is blocked for this site',
    )
    await page.getByTestId('mic-help').getByRole('button', { name: 'Dismiss' }).click()
    await expect(page.getByTestId('mic-help')).toHaveCount(0)
    // And the attempt itself is never blocked by the page.
    await page.getByTestId('hero-mic').click()
    const help = page.getByTestId('mic-help')
    await expect(help).toContainText('NotAllowedError: Permission denied')
    await expect(help.getByTestId('mic-retry')).toBeVisible()
    await expect(help.getByTestId('mic-help-sample')).toBeVisible()
    await expect(help.getByTestId('mic-help-upload')).toBeVisible()
    await axe(page)
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    )
    expect(overflow).toBeLessThanOrEqual(1)
  })

  test('the banner dot stays on the first line of its text at 390 px', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await open(page, 'stt')
    const banner = page.getByTestId('demo-banner')
    const dot = await banner.locator('span[aria-hidden]').first().boundingBox()
    const text = await banner.getByText('Live demo.').boundingBox()
    if (!dot || !text) throw new Error('banner not laid out')
    // Same line: the dot's centre lies within the first line of text, and to its left.
    const centre = dot.y + dot.height / 2
    expect(centre).toBeGreaterThan(text.y)
    expect(centre).toBeLessThan(text.y + text.height)
    expect(dot.x + dot.width).toBeLessThanOrEqual(text.x)
  })

  test('with data saver on, nothing is fetched until asked', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'connection', {
        configurable: true,
        value: { saveData: true, effectiveType: '4g' },
      })
    })
    const modelRequests: string[] = []
    page.on('request', (r) => {
      if (/\.onnx|ort-wasm/.test(r.url())) modelRequests.push(r.url())
    })
    await open(page, 'stt')
    await expect(page.getByTestId('data-saver-note')).toContainText('not fetched in advance')
    await page.waitForTimeout(4000)
    await expect(page.getByTestId('demo-hero').getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')
    expect(modelRequests).toEqual([])
  })

  test('?debug=1 reports the service worker and isolation', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('bolo.theme', 'dark'))
    await page.goto('./?debug=1#/stt')
    await expect.poll(() => page.evaluate(() => crossOriginIsolated), { timeout: 20_000 }).toBe(true)
    const env = page.getByTestId('diagnostics-env')
    await expect(env).toContainText(/crossOriginIsolated\s*yes/)
    await expect(env).toContainText('credentialless (added by the service worker)')
    await expect(env).toContainText('controlling this page')
    await expect(env).toContainText(/SharedArrayBuffer\s*available/)
  })
})

test.describe('with the models available: first use and Hindi', () => {
  test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a local model mirror to run')

  test('the model is fetched while the page is idle, with real stages, before anything is pressed', async ({
    page,
  }) => {
    await open(page, 'stt')
    const chip = page.getByTestId('demo-hero').getByTestId('model-chip')
    await expect(chip).toHaveAttribute('data-status', /loading|ready/, { timeout: 20_000 })
    await expect(chip).toContainText('Ready', { timeout: 120_000 })
    // Then talking starts at once.
    await page.getByTestId('hero-mic').click()
    await expect(page.getByTestId('stop')).toBeVisible()
    await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 60_000 })
    await page.getByTestId('stop').click()
  })

  test('tapping the microphone before the model is there: live at once, speech kept, then transcribed', async ({
    page,
  }) => {
    // Hold the model back so the wait is long enough to observe.
    await page.route('**/onnx/*.onnx', async (route) => {
      await new Promise((r) => setTimeout(r, 4000))
      await route.continue()
    })
    await open(page, 'stt')
    await page.getByTestId('hero-mic').click()
    const note = page.getByTestId('starting-note')
    await expect(note).toHaveAttribute('data-mic', 'live', { timeout: 10_000 })
    await expect(page.getByTestId('load-stage').first()).toContainText(/Downloading|Compiling|Warming/)
    await expect
      .poll(
        async () =>
          Number.parseFloat(
            await page
              .getByTestId('buffered-seconds')
              .innerText()
              .catch(() => '99'),
          ),
        {
          timeout: 20_000,
        },
      )
      .toBeGreaterThan(1.5)
    await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 150_000 })
    // The fake microphone plays the Kennedy sample from its start: the transcript begins with its opening words,
    // which were spoken while the model was still loading.
    expect((await page.getByTestId('transcript').innerText()).trim()).toMatch(/^and so,? my fellow america/i)
    await page.getByTestId('stop').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 60_000 })
  })

  test('Hindi: the fine-tuned model transcribes the bundled Hindi sample in Devanagari', async ({ page }) => {
    await page.addInitScript(() => {
      if (!localStorage.getItem('bolo.prefs'))
        localStorage.setItem(
          'bolo.prefs',
          JSON.stringify({ browserModel: 'whisper-tiny-hindi', browserModelPicked: true }),
        )
    })
    await open(page, 'stt')
    await expect(page.getByTestId('browser-model-picker')).toContainText('Whisper Tiny Hindi')
    await expect(page.getByTestId('language-select')).toContainText('हिन्दी · Hindi')
    const t0 = Date.now()
    await page.getByTestId('sample-hi-short').click()
    const engine = page.getByTestId('browser-engine')
    await expect(engine.getByTestId('engine-status')).toHaveAttribute('data-status', 'ready', { timeout: 180_000 })
    const readyMs = Date.now() - t0
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 120_000 })
    const transcript = await page.getByTestId('transcript').innerText()
    expect(transcript).toContain('नमस्ते')
    expect(transcript).toMatch(/मौसम|अच्छा/)
    const p50 = await page.getByTestId('readout').filter({ hasText: 'Final p50' }).innerText()
    const rtf = await page.getByTestId('readout').filter({ hasText: 'Real-time factor' }).innerText()
    console.log(
      `[demo-hindi] ready ${readyMs} ms · ${p50.replace(/\s+/g, ' ')} · ${rtf.replace(/\s+/g, ' ')} · ${transcript.trim()}`,
    )
    await axe(page)
  })
})

test.describe('downloaded models', () => {
  const seeded = {
    entries: [
      {
        id: 'syspin-mai-female',
        kind: 'tts',
        label: 'SYSPIN Maithili, female',
        bytes: 36_500_000,
        cache: 'bolo-voices',
        keys: [],
        addedAt: 1,
        lastUsed: 3,
      },
      {
        id: 'syspin-hi-female',
        kind: 'tts',
        label: 'SYSPIN Hindi, female',
        bytes: 36_500_000,
        cache: 'bolo-voices',
        keys: [],
        addedAt: 1,
        lastUsed: 1,
      },
      {
        id: 'whisper-base-q8',
        kind: 'stt',
        label: 'Whisper Base',
        bytes: 79_700_000,
        cache: 'transformers-cache',
        keys: [],
        addedAt: 1,
        lastUsed: 2,
      },
    ],
    capBytes: null,
  }

  test('both screens list every downloaded model with its size; delete, limit and clear all', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 900 })
    await page.addInitScript((s) => {
      if (!sessionStorage.getItem('seeded')) {
        localStorage.setItem('bolo.downloads', s)
        sessionStorage.setItem('seeded', '1')
      }
    }, JSON.stringify(seeded))
    await open(page, 'tts')
    const panel = page.getByTestId('downloads-panel')
    await expect(panel).toContainText('3 models')
    await expect(panel.getByTestId('download-whisper-base-q8')).toContainText('Speech to text')
    await expect(panel.getByTestId('download-syspin-mai-female')).toContainText('36.5 MB')
    await expect(panel.getByTestId('storage-use')).toContainText('This site stores')
    await page.getByTestId('delete-syspin-mai-female').click()
    await expect(panel).toContainText('2 models')
    // A limit below what is stored removes the least recently used first.
    await page.getByTestId('storage-cap').click()
    await page.getByRole('option', { name: '250 MB' }).click()
    await expect(panel).toContainText('2 models')
    await page.getByTestId('storage-cap').click()
    await page.getByRole('option', { name: 'No limit' }).click()
    await expect(page.getByRole('listbox')).toHaveCount(0)
    await page.waitForTimeout(400)
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    )
    expect(overflow).toBeLessThanOrEqual(1)
    await axe(page)

    await page.goto('./#/stt')
    await expect(page.getByTestId('downloads-panel')).toContainText('Whisper Base')
    await page.getByTestId('downloads-panel').getByTestId('clear-downloads').click()
    await expect(page.getByTestId('downloads-panel')).toContainText('Nothing yet')
  })

  test('after the page closed mid-load, nothing loads by itself and clearing is offered', async ({ page }) => {
    await page.addInitScript(() => {
      if (!sessionStorage.getItem('seeded')) {
        localStorage.setItem('bolo.loadingModel', JSON.stringify({ kind: 'stt', id: 'whisper-base-q8', at: 1 }))
        sessionStorage.setItem('seeded', '1')
      }
    })
    const modelRequests: string[] = []
    page.on('request', (r) => {
      if (/\.onnx|ort-wasm/.test(r.url())) modelRequests.push(r.url())
    })
    await open(page, 'stt')
    const notice = page.getByTestId('interrupted-load')
    await expect(notice).toContainText('The page closed while loading Whisper Base')
    await page.waitForTimeout(5000)
    expect(modelRequests).toEqual([])
    await expect(page.getByTestId('demo-hero').getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')
    await axe(page)
    await page.getByTestId('interrupted-clear').click()
    await expect(notice).toHaveCount(0)
  })
})

test.describe('text to speech in the page', () => {
  test('the voice picker lists every language by its own name, with size and licence', async ({ page }) => {
    await open(page, 'tts')
    await expect(page.getByTestId('needs-server')).toHaveCount(0)
    await expect(page.getByTestId('tts-text')).toHaveValue('नमस्ते! आप कैसे हैं? आज मौसम बहुत अच्छा है।')
    await page.getByTestId('voice-picker').click()
    for (const name of [
      'हिन्दी · Hindi',
      'मैथिली · Maithili',
      'भोजपुरी · Bhojpuri',
      'मराठी · Marathi',
      'ಕನ್ನಡ · Kannada',
      'English (India)',
      'English (US)',
    ]) {
      await expect(page.getByRole('group').filter({ hasText: name }).first()).toBeVisible()
    }
    await expect(page.getByTestId('voice-syspin-mai-female')).toContainText('MB')
    await page.getByTestId('voice-syspin-mai-female').click()
    // A sample sentence is swapped for the new language's own.
    await expect(page.getByTestId('tts-text')).toHaveValue('प्रणाम! अहाँ केहन छी? आइ मौसम बड्ड नीक अछि।')
    const about = page.getByTestId('voice-about')
    await expect(about).toContainText('SYSPIN project, SPIRE Lab, Indian Institute of Science')
    await expect(about).toContainText('MIT')
    await expect(about).toContainText('CC-BY-4.0')
    await expect(about).toContainText('Nobody who speaks this language has reviewed')
    // No in-browser recogniser for Maithili: the round trip says so instead of failing.
    await expect(page.getByTestId('round-trip')).toBeDisabled()
    if (!VOICES) await expect(page.getByTestId('voices-missing')).toBeVisible()
  })

  test.describe('with the voices published', () => {
    test.skip(!VOICES || !MIRROR, 'set BOLO_E2E_VOICE_DIR (voices) and BOLO_E2E_MODEL_DIR (runtime mirror) to run')

    test('speaks Maithili sentence by sentence and measures it', async ({ page }) => {
      await open(page, 'tts')
      await expect(page.getByTestId('voices-missing')).toHaveCount(0)
      await page.getByTestId('voice-picker').click()
      await page.getByTestId('voice-syspin-mai-female').click()
      await page.getByTestId('tts-text').fill('प्रणाम! काल्हि 10 बजे बैसक अछि।')
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 })
      await expect(page.getByTestId('ttfa')).toHaveText(/^\d+$/, { timeout: 60_000 })
      // Numbers are spelt out, and the page says what it read.
      await expect(page.getByTestId('spoken-text')).toContainText('दस बजे')
      await expect(page.getByTestId('chunk-timeline')).toBeVisible()
      await expect(page.getByTestId('speak')).toBeVisible({ timeout: 60_000 })
      // The one-word greeting is not synthesised by itself (the voice garbles it): it goes with the next sentence.
      await expect(page.getByTestId('chunk-timeline')).toContainText('प्रणाम! काल्हि दस बजे बैसक अछि।')
      await expect(page.getByTestId('save-wav')).toBeEnabled()
      const download = page.waitForEvent('download')
      await page.getByTestId('save-wav').click()
      expect((await download).suggestedFilename()).toBe('bolo-syspin-mai-female.wav')
      const readouts = await page.getByTestId('readout').allInnerTexts()
      console.log(
        `[tts] mai ttfa ${await page.getByTestId('ttfa').innerText()} ms · ${readouts.join(' · ').replace(/\n/g, ' ')}`,
      )
      await axe(page)
    })

    test('out of memory (as on an iPhone): one lean retry, then plain advice', async ({ page }) => {
      // Stand-in for WebKit's limit: no shared WebAssembly memory above 512 MiB in the synthesiser's worker.
      page.on('worker', (w) => {
        void w
          .evaluate(() => {
            if (!location.pathname.includes('/assets/worker-')) return
            const Real = WebAssembly.Memory
            const Limited = function (d: WebAssembly.MemoryDescriptor & { shared?: boolean }) {
              if (d.shared && (d.maximum ?? 0) > 8192) throw new RangeError('Out of memory')
              return new Real(d)
            } as unknown as typeof WebAssembly.Memory
            Limited.prototype = Real.prototype
            WebAssembly.Memory = Limited
          })
          .catch(() => undefined)
      })
      await open(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId('voice-piper-en-us-amy-low').click()
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('load-oom')).toContainText('Close other tabs', { timeout: 120_000 })
      await expect(page.getByTestId('load-trouble')).toContainText('already retried on one thread')
      await axe(page)
      // The way out: unload everything and remove the downloads, with no reload loop.
      await page.getByTestId('oom-clear').click()
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')
      await expect(page.getByTestId('downloads-panel')).toContainText('Nothing yet')
    })

    test('on an iPhone only one model runtime is kept: the voice gives way to the recogniser and back', async ({
      browser,
    }) => {
      const context = await browser.newContext({
        userAgent:
          'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 EdgiOS/141.0 Mobile/15E148 Safari/605.1.15',
        viewport: { width: 390, height: 844 },
      })
      const page = await context.newPage()
      await open(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId('voice-piper-en-us-amy-low').click()
      await page.getByTestId('tts-text').fill('Good morning.')
      await page.getByTestId('round-trip').click()
      await expect(page.getByTestId('wer')).toBeVisible({ timeout: 180_000 })
      // The recogniser needed the memory: the voice was released and says it loads again when needed.
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')
      await page.getByTestId('speak').click()
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 })
      await context.close()
    })

    test('switching models keeps one voice and one recogniser loaded; nothing loads at start-up', async ({ page }) => {
      const live = new Set<string>()
      page.on('worker', (w) => {
        live.add(w.url())
        w.on('close', () => live.delete(w.url()))
      })
      const modelWorkers = () => page.workers().filter((w) => /\/assets\/worker-/.test(w.url())).length
      await open(page, 'tts')
      for (const id of ['syspin-mai-male', 'piper-en-us-amy-low', 'syspin-hi-female']) {
        await page.getByTestId('voice-picker').click()
        await page.getByTestId(`voice-${id}`).click()
        await page.getByTestId('tts-text').fill(id.startsWith('piper') ? 'Good morning.' : 'नमस्ते।')
        await page.getByTestId('speak').click()
        await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 })
        await expect(page.getByTestId('speak')).toBeVisible({ timeout: 60_000 })
        // The previous voice's worker was terminated: one model worker, whatever was loaded before.
        await expect.poll(modelWorkers).toBe(1)
      }
      const panel = page.getByTestId('downloads-panel')
      for (const id of ['syspin-mai-male', 'piper-en-us-amy-low', 'syspin-hi-female'])
        await expect(panel.getByTestId(`download-${id}`)).toContainText('MB')
      await expect(panel.getByTestId('download-syspin-hi-female')).toContainText('loaded now')

      // A reload with three voices downloaded starts nothing.
      await page.reload()
      await expect(page.getByTestId('downloads-panel')).toContainText('3 models')
      await page.waitForTimeout(4000)
      expect(modelWorkers()).toBe(0)
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')

      // Delete one: it goes from the list and from the cache, by its own keys.
      await page.getByTestId('delete-syspin-mai-male').click()
      await expect(page.getByTestId('download-syspin-mai-male')).toHaveCount(0)
      const left = await page.evaluate(async () =>
        Promise.all(
          ['syspin-mai-male', 'syspin-hi-female'].map(async (id) => {
            const c = await caches.open('bolo-voices')
            const hit = await c.match(new URL(`voices/${id}/voice.json`, location.href).href, { ignoreSearch: true })
            return Boolean(hit)
          }),
        ),
      )
      expect(left).toEqual([false, true])
      await page.getByTestId('clear-downloads').click()
      await expect(page.getByTestId('downloads-panel')).toContainText('Nothing yet')
    })

    test('on an Android phone only one model runtime is kept in total', async ({ browser }) => {
      const context = await browser.newContext({
        userAgent:
          'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36',
        viewport: { width: 390, height: 844 },
      })
      const page = await context.newPage()
      await open(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId('voice-piper-en-us-amy-low').click()
      await page.getByTestId('tts-text').fill('Good morning.')
      await page.getByTestId('round-trip').click()
      await expect(page.getByTestId('wer')).toBeVisible({ timeout: 180_000 })
      await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'idle')
      expect(page.workers().filter((w) => /\/assets\/worker-/.test(w.url())).length).toBe(1)
      await context.close()
    })

    test('speaks US English through uccharan and transcribes it back in the page', async ({ page }) => {
      await open(page, 'tts')
      await page.getByTestId('voice-picker').click()
      await page.getByTestId('voice-piper-en-us-amy-low').click()
      await page.getByTestId('tts-text').fill('Hello there. This voice runs entirely in your browser.')
      await page.getByTestId('round-trip').click()
      await expect(page.getByTestId('wer')).toBeVisible({ timeout: 180_000 })
      await expect(page.getByTestId('phonemes')).toContainText('həlˈoʊ')
      const wer = await page.getByTestId('wer').innerText()
      console.log(`[tts] amy round trip WER ${wer}: ${await page.getByTestId('wer-diff').innerText()}`)
      expect(Number.parseFloat(wer)).toBeLessThanOrEqual(30)
    })
  })
})
