import { expect, test } from '@playwright/test'
import { open, watchErrors } from './helpers'

test.use({ viewport: { width: 1440, height: 900 } })

test.describe('Models', () => {
  test('search, filter, table view and the details drawer', async ({ page }) => {
    await open(page, '/models')
    await expect(page.getByTestId('model-card').first()).toBeVisible()
    const all = await page.getByTestId('model-card').count()
    await page.getByLabel('Search models').fill('whisper')
    await expect(page.getByTestId('model-card')).toHaveCount(3)
    await page.getByRole('button', { name: 'Clear' }).click()
    await expect(page.getByTestId('model-card')).toHaveCount(all)
    await page.getByRole('radio', { name: 'TTS' }).click()
    await expect(page.getByRole('heading', { name: /Text to speech/ })).toBeVisible()
    await expect(page.getByRole('heading', { name: /Speech to text/, level: 2 })).toHaveCount(0)
    await page.getByRole('combobox', { name: 'Language' }).click()
    await page.getByRole('option', { name: 'Hindi' }).click()
    await expect(page.getByTestId('model-card').filter({ hasText: 'piper-hi-in-pratham' })).toBeVisible()
    await expect(page.getByTestId('model-card').filter({ hasText: 'piper-en-us-lessac' })).toHaveCount(0)

    await page.getByRole('radio', { name: 'Table' }).click()
    await expect(page.getByTestId('model-row').first()).toBeVisible()
    await page.getByRole('button', { name: 'kokoro-82m', exact: true }).click()
    const drawer = page.getByRole('dialog')
    await expect(drawer.getByRole('heading', { name: 'kokoro-82m' })).toBeVisible()
    await expect(drawer.getByRole('heading', { name: 'Measured on this server' })).toBeVisible()
    await expect(drawer.getByText('Real-time factor p50')).toBeVisible()
    await expect(drawer.getByText('Validated on CPU')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(drawer).toBeHidden()
  })

  test('pull with live progress, then load, unload and delete', async ({ page }) => {
    await open(page, '/models')
    const card = page.getByTestId('model-card').filter({ hasText: 'ten-vad' })
    await card.getByTestId('pull-ten-vad').click()
    await expect(page.getByText('Pulling ten-vad')).toBeVisible()
    await expect(card.getByText('Installed', { exact: true })).toBeVisible({ timeout: 15_000 })
    await card.getByTestId('load-ten-vad').click()
    await expect(card.getByText('Loaded', { exact: true })).toBeVisible({ timeout: 10_000 })
    await card.getByRole('button', { name: 'Unload' }).click()
    await expect(card.getByText('Installed', { exact: true })).toBeVisible()
    await card.getByRole('button', { name: 'More actions for ten-vad' }).click()
    await page.getByRole('menuitem', { name: 'Delete from disk' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click()
    await expect(card.getByText('Not installed')).toBeVisible()
  })

  test('progress is shown while a larger model downloads, and gated models are refused clearly', async ({ page }) => {
    await open(page, '/models')
    const card = page.getByTestId('model-card').filter({ hasText: 'whisper-small' })
    await card.getByTestId('pull-whisper-small').click()
    await expect(card.getByRole('progressbar')).toBeVisible()
    await expect(card.getByText(/^\d+%$/)).toBeVisible()
    const gated = page.getByTestId('model-card').filter({ hasText: 'indic-parler-tts' })
    await expect(gated.getByText('Gated', { exact: true })).toBeVisible()
    await gated.getByTestId('pull-indic-parler-tts').click()
    await expect(page.getByText('Could not pull indic-parler-tts')).toBeVisible()
  })
})

test.describe('Latency', () => {
  test('live tiles, charts, waterfall drill-down, filters and pause', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/latency')
    await expect(page.getByTestId('stat-tile').filter({ hasText: 'Active streams' })).toContainText(/\d+/)
    await expect(page.getByTestId('latency-series')).toBeVisible()
    await expect(page.getByTestId('latency-histogram')).toBeVisible()
    const rows = page.getByTestId('waterfall').getByRole('button')
    await expect(rows.first()).toBeVisible()

    await rows.first().click()
    const drawer = page.getByTestId('request-drawer')
    await expect(drawer.getByRole('heading', { name: /^r_/ })).toBeVisible()
    await expect(drawer.getByRole('row', { name: /Queue/ })).toBeVisible()
    await expect(drawer.getByText('Real-time factor')).toBeVisible()
    await page.keyboard.press('Escape')

    // Model filter scopes the waterfall.
    await page.getByRole('combobox', { name: 'Model' }).click()
    await page.getByRole('option', { name: 'kokoro-82m' }).click()
    await expect(rows.first()).toContainText('kokoro-82m')
    await expect(page.getByTestId('waterfall')).not.toContainText('moonshine-tiny-en')
    await page.getByRole('combobox', { name: 'Latency metric' }).click()
    await page.getByRole('option', { name: 'Time to first audio' }).click()
    await expect(page.getByRole('heading', { name: 'Time to first audio over time' })).toBeVisible()

    // Pause freezes the feed; resume catches up.
    await page.getByTestId('pause').click()
    await expect(page.getByText(/Paused at/)).toBeVisible()
    const frozen = await rows.first().innerText()
    await page.waitForTimeout(3500)
    expect(await rows.first().innerText()).toBe(frozen)
    await page.getByTestId('pause').click()
    await expect(page.getByText(/Paused at/)).toBeHidden()

    // Every chart has a table twin.
    await page.getByRole('button', { name: /Show Time to first audio over time as a table/ }).click()
    await expect(page.getByRole('columnheader', { name: 'p95 ms' })).toBeVisible()
    expect(errors).toEqual([])
  })

  test('time range toggle changes the bucket size', async ({ page }) => {
    await open(page, '/latency')
    await expect(page.getByText('Percentiles per 10 s bucket')).toBeVisible()
    await page.getByRole('radio', { name: '1 min' }).click()
    await expect(page.getByText('Percentiles per 2 s bucket')).toBeVisible()
  })
})

test.describe('Benchmark', () => {
  test('runs a sweep, marks the ceiling, exports and compares', async ({ page }) => {
    await open(page, '/benchmark')
    await expect(page.getByText('No benchmark yet')).toBeVisible()
    await page.getByRole('button', { name: 'up to 16' }).click()
    await page.getByTestId('bench-run').click()
    await expect(page.getByRole('status').filter({ hasText: 'Benchmarking' })).toBeVisible()
    await expect(page.getByText('Benchmark finished')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('ceiling')).toContainText(/Concurrency ceiling/)
    await expect(page.getByTestId('ceiling').locator('.text-5xl')).toHaveText(/^\d+$/)
    await expect(page.getByTestId('bench-table').locator('tbody tr')).toHaveCount(5)
    await expect(page.getByTestId('bench-latency')).toBeVisible()
    await expect(page.getByTestId('bench-throughput')).toBeVisible()

    await page.getByRole('button', { name: 'Export' }).click()
    const csv = page.waitForEvent('download')
    await page.getByRole('menuitem', { name: 'CSV' }).click()
    const file = await csv
    expect(file.suggestedFilename()).toMatch(/^bolo-bench-moonshine-tiny-en-.*\.csv$/)

    // A second run on another model, then compare the two.
    await page.getByTestId('model-picker').click()
    await page.getByRole('option', { name: /zipformer-streaming-en/ }).click()
    await page.getByTestId('bench-run').click()
    await expect(page.getByText('Benchmark finished').last()).toBeVisible({ timeout: 30_000 })
    await page.getByRole('combobox', { name: 'Compare with another run' }).click()
    await page.getByRole('option', { name: /moonshine-tiny-en/ }).click()
    await expect(page.getByTestId('ceiling')).toContainText(/vs \d+ for moonshine-tiny-en/)

    // Runs survive a reload.
    await page.reload()
    await expect(page.getByRole('button', { name: /zipformer-streaming-en/ }).first()).toBeVisible()
  })

  test('rejects an empty concurrency list', async ({ page }) => {
    await open(page, '/benchmark')
    await page.getByLabel('Concurrency levels').fill('abc')
    await expect(page.getByText('Enter numbers or ranges')).toBeVisible()
    await expect(page.getByTestId('bench-run')).toBeDisabled()
  })
})

test.describe('Connect', () => {
  test('snippets carry this server address and the chosen models, and copy works', async ({ page }) => {
    await open(page, '/connect')
    const origin = new URL(page.url()).origin
    await expect(page.locator('pre').first()).toContainText(`${origin}/v1/audio/transcriptions`)
    await expect(page.locator('pre').first()).toContainText('model=moonshine-tiny-en')
    await page.getByRole('tab', { name: 'Python' }).click()
    await expect(page.locator('pre').first()).toContainText(`base_url="${origin}/v1"`)
    await page.getByRole('button', { name: /^Copy: Transcribe and speak/ }).click()
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('from openai import OpenAI')
    await page.getByRole('tab', { name: 'WebSocket' }).click()
    await expect(page.locator('pre').first()).toContainText(`${origin.replace('http', 'ws')}/api/v1/stt/stream`)
    await page.getByRole('tab', { name: 'Home Assistant' }).click()
    await expect(page.locator('pre').first()).toContainText('Port: 10300')
    await expect(page.getByText('sherpa-onnx')).toBeVisible()
    await expect(page.getByTestId('support-list')).toContainText('WebAssembly SIMD')
    await expect(page.getByRole('link', { name: /Docs/ })).toHaveAttribute('href', /github\.com\/cb-Sigma\/bolo/)
  })
})
