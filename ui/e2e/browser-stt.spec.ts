import fs from 'node:fs'
import path from 'node:path'
import { expect, test } from '@playwright/test'
import { open, watchErrors } from './helpers'

/**
 * Runs the in-browser recogniser for real: ONNX Runtime Web (WebAssembly)
 * with Moonshine Tiny, transcribing the built-in speech sample.
 *
 * The model is ~28 MB, so this only runs when a local copy is provided:
 *
 *   BOLO_E2E_MODEL_DIR=/path/to/mirror npm run e2e -- browser-stt
 *
 * where the directory mirrors the Hugging Face layout
 * (`onnx-community/moonshine-tiny-ONNX/resolve/main/...`). Requests to
 * huggingface.co and the jsDelivr copy of the ONNX runtime are answered from
 * disk, so the page runs exactly the code path a user gets, without the network.
 */
const MIRROR = process.env.BOLO_E2E_MODEL_DIR
const ORT_DIST = path.resolve(import.meta.dirname, '../node_modules/onnxruntime-web/dist')
const cors = { 'access-control-allow-origin': '*', 'cross-origin-resource-policy': 'cross-origin' }

test.describe('in-browser speech recognition (WebAssembly)', () => {
  test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a local model mirror to run')
  test.use({ viewport: { width: 1440, height: 1000 } })
  test.setTimeout(180_000)

  test.beforeEach(async ({ context }) => {
    await context.route('https://huggingface.co/**', (route) => {
      const file = path.join(MIRROR as string, new URL(route.request().url()).pathname)
      if (!fs.existsSync(file)) return route.fulfill({ status: 404, headers: cors, body: 'not mirrored' })
      return route.fulfill({
        status: 200,
        headers: { ...cors, 'content-type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream' },
        body: fs.readFileSync(file),
      })
    })
    await context.route('https://cdn.jsdelivr.net/**', (route) => {
      const file = path.join(ORT_DIST, path.basename(new URL(route.request().url()).pathname))
      if (!fs.existsSync(file)) return route.fulfill({ status: 404, headers: cors })
      return route.fulfill({
        status: 200,
        headers: { ...cors, 'content-type': file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' },
        body: fs.readFileSync(file),
      })
    })
  })

  test('downloads the model, transcribes a sample with no server round trip, and caches it', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/stt')
    await page.getByTestId('mode-browser').click()
    const panel = page.getByTestId('browser-engine')
    await expect(panel).toContainText('Moonshine Tiny')
    await expect(panel.getByTestId('engine-status')).toHaveText('Not downloaded')

    const started = Date.now()
    await page.getByTestId('sample-en-ask-not').click()
    await expect(panel.getByTestId('engine-status')).toHaveAttribute('data-status', 'ready', { timeout: 120_000 })
    const loadMs = Date.now() - started
    await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 60_000 })
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 60_000 })

    const transcript = (await page.getByTestId('transcript').innerText()).toLowerCase()
    expect(transcript).toContain('my fellow americans')
    expect(transcript).toContain('what you can do for your country')
    await expect(panel).toContainText(/WebAssembly · \d thread/)
    await expect(page.getByTestId('latency-timeline').getByRole('img')).toHaveAccessibleName(
      /final \d[\d,]* ms after speech ended/,
    )
    // No network: the split attributes nothing to it.
    await expect(page.getByRole('img', { name: /Latency split/ })).toHaveAccessibleName(/Network 0 ms/)

    const info = await page.evaluate(() => ({ isolated: crossOriginIsolated, cores: navigator.hardwareConcurrency }))
    const finalP50 = await page.getByTestId('readout').filter({ hasText: 'Final p50' }).innerText()
    const rtf = await page.getByTestId('readout').filter({ hasText: 'Real-time factor' }).innerText()
    console.log(
      `[browser-stt] model ready in ${loadMs} ms · ${JSON.stringify(info)} · ${finalP50.replace(/\s+/g, ' ')} · ${rtf.replace(/\s+/g, ' ')}`,
    )

    // Second visit: weights come from Cache Storage.
    await page.reload()
    await page.getByTestId('mode-browser').click()
    await expect(page.getByTestId('browser-engine').getByTestId('engine-status')).toHaveText('Cached')
    expect(errors.filter((e) => !/adapter/i.test(e))).toEqual([])
  })

  test('compare mode runs the same audio through the server and the browser side by side', async ({ page }) => {
    await open(page, '/stt')
    await page.getByTestId('mode-compare').click()
    await page.getByTestId('sample-en-fellow-americans').click()
    await expect(page.getByTestId('browser-engine').getByTestId('engine-status')).toHaveAttribute(
      'data-status',
      'ready',
      { timeout: 120_000 },
    )
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 60_000 })
    const transcripts = page.getByTestId('transcript')
    await expect(transcripts).toHaveCount(2)
    await expect(transcripts.nth(0)).toContainText(/fellow americans/i)
    await expect(transcripts.nth(1)).toContainText(/fellow.americans/i)
    await expect(page.getByTestId('latency-timeline')).toHaveCount(2)
    await expect(page.getByTestId('budget-bar')).toHaveCount(2)
  })
})
