import fs from 'node:fs'
import path from 'node:path'
import { expect, test, type Browser, type Page } from '@playwright/test'
import { BOLO_PORT, MODELS, SAMPLE } from '../playwright.webrtc.config'

/**
 * A voice-agent turn over WebRTC in headless Chromium against a real `bolo serve` (see playwright.webrtc.config.ts):
 * the fake microphone's speech must come back as a transcript event on the data channel, and the answer must
 * arrive as sound on the remote audio track. Once through the console's Conversation page, once through
 * examples/webrtc_browser.html.
 */

const server = `http://127.0.0.1:${BOLO_PORT}`
const example = path.resolve(import.meta.dirname, '../../examples/webrtc_browser.html')
const SAMPLE_TEXT = 'Good morning. Could you tell me when my order will arrive?'

interface Run {
  page: string
  connected_ms: number
  user_transcript_ms: number
  agent_text_ms: number | null
  agent_sound_ms: number
  user_transcript: string
  agent_text: string | null
  remote_audio: { packets: number; seconds: number } | null
  events_seen: string[]
  server_trace: Record<string, unknown> | null
}

const runs: Run[] = []
let browserVersion = ''

test.beforeAll(async ({ browser }: { browser: Browser }) => {
  browserVersion = `${browser.browserType().name()} ${browser.version()}`
})

/** The newest voice-agent turn the server traced over WebRTC: models, phases and latency as bolo saw them. */
async function serverTrace(page: Page): Promise<Record<string, unknown> | null> {
  const resp = await page.request.get(`${server}/api/v1/requests`, { params: { kind: 'agent', limit: 20 } })
  const body = (await resp.json()) as { requests?: Record<string, unknown>[] } | Record<string, unknown>[]
  const list = Array.isArray(body) ? body : (body.requests ?? [])
  const t = list.find((r) => r.transport === 'webrtc')
  if (!t) return null
  const phases = Object.fromEntries(
    ((t.phases as { name: string; dur_ms: number }[]) ?? []).map((p) => [p.name, Math.round(p.dur_ms)]),
  )
  const pick = (k: string) => (typeof t[k] === 'number' ? Math.round(t[k] as number) : (t[k] ?? null))
  return {
    transport: t.transport,
    status: t.status,
    stt_model: t.stt_model ?? null,
    model: t.model,
    final_ms: pick('final_ms'),
    first_token_ms: pick('first_token_ms'),
    ttfa_ms: pick('ttfa_ms'),
    total_ms: pick('total_ms'),
    phases,
  }
}

test('the Conversation page talks over WebRTC', async ({ page }) => {
  await page.goto('/#/conversation')
  await page.getByTestId('transport-webrtc').click()
  await expect(page.getByTestId('transport-webrtc')).toHaveAttribute('data-state', 'on')
  const t0 = Date.now()
  await page.getByTestId('start-call').click()
  const state = page.getByTestId('webrtc-state')
  await expect(state).toHaveAttribute('data-connection', 'connected')
  const connected = Date.now() - t0
  await expect(page.getByText('Live · pipeline cascade')).toBeVisible()
  // the fake microphone's words, recognised by the server and sent back as a transcript event
  const user = page.locator('[data-testid="turns"] li').filter({ hasText: /order/i }).first()
  await expect(user).toBeVisible({ timeout: 90_000 })
  const transcribed = Date.now() - t0
  const agent = page
    .locator('[data-testid="turns"] li')
    .filter({ hasText: /You said/ })
    .first()
  await expect(agent).toBeVisible()
  const answered = Date.now() - t0
  // and the answer as sound on the remote track
  await expect(state).toHaveAttribute('data-heard', 'true')
  const heard = Date.now() - t0
  const stateText = (await state.textContent()) ?? ''
  const userText = ((await user.textContent()) ?? '').replace(/^You/, '').trim()
  const agentText = ((await agent.textContent()) ?? '').replace(/^Agent\s*(first audio [\d,.]+\s*m?s)?/, '').trim()
  await page.getByTestId('end-call').click()
  await expect(page.getByText(/^Ended/)).toBeVisible()
  await expect(page.getByTestId('conversation-error')).toHaveCount(0)
  const seconds = /agent audio ([\d.]+) s/.exec(stateText)?.[1]
  const packets = Number(await state.getAttribute('data-packets'))
  expect(packets).toBeGreaterThan(50)
  runs.push({
    page: 'console Conversation page (transport WebRTC)',
    connected_ms: connected,
    user_transcript_ms: transcribed,
    agent_text_ms: answered,
    agent_sound_ms: heard,
    user_transcript: userText,
    agent_text: agentText,
    remote_audio: seconds ? { packets, seconds: Number(seconds) } : null,
    events_seen: [],
    server_trace: await serverTrace(page),
  })
})

test('examples/webrtc_browser.html talks over WebRTC', async ({ page }) => {
  // served from the server's own origin, as if bolo hosted it, so no CORS setting is needed
  await page.route(`${server}/webrtc_browser.html`, (route) =>
    route.fulfill({ contentType: 'text/html', body: fs.readFileSync(example, 'utf8') }),
  )
  await page.goto(`${server}/webrtc_browser.html`)
  await expect(page.locator('#server')).toHaveValue(server)
  const t0 = Date.now()
  await page.locator('#start').click()
  await expect(page.getByTestId('status')).toHaveText('live')
  const connected = Date.now() - t0
  await expect(page.locator('#channel')).toHaveText('open')
  type Call = {
    state: string
    events: string[]
    user: string[]
    assistant: string[]
    audio: { packets: number; seconds: number; energy: number } | null
    error: string | null
  }
  const call = () => page.evaluate(() => (window as unknown as { boloCall: Call }).boloCall)
  let transcribed = 0
  let answered: number | null = null
  await expect
    .poll(
      async () => {
        const c = await call()
        if (c.error) throw new Error(c.error)
        if (c.user.length && !transcribed) transcribed = Date.now() - t0
        if (c.assistant.length && answered == null) answered = Date.now() - t0
        return c.user.length > 0
      },
      { timeout: 90_000, intervals: [200] },
    )
    .toBe(true)
  await expect.poll(async () => ((await call()).audio?.energy ?? 0) > 1e-4, { timeout: 60_000 }).toBe(true)
  const heard = Date.now() - t0
  await expect.poll(async () => (await call()).assistant.length, { timeout: 30_000 }).toBeGreaterThan(0)
  if (answered == null) answered = Date.now() - t0
  const c = await call()
  expect(c.user[0]).toMatch(/order/i)
  expect(c.events).toContain('output_audio_buffer.started')
  expect(c.audio?.packets).toBeGreaterThan(50)
  await expect(page.getByTestId('user-turn').first()).toContainText(/order/i)
  await page.locator('#stop').click()
  await expect(page.getByTestId('status')).toHaveText('ended')
  runs.push({
    page: 'examples/webrtc_browser.html',
    connected_ms: connected,
    user_transcript_ms: transcribed,
    agent_text_ms: answered,
    agent_sound_ms: heard,
    user_transcript: c.user[0] ?? '',
    agent_text: c.assistant[0] ?? null,
    remote_audio: c.audio ? { packets: c.audio.packets, seconds: Math.round(c.audio.seconds * 10) / 10 } : null,
    events_seen: [...new Set(c.events)].sort(),
    server_trace: await serverTrace(page),
  })
})

test.afterAll(() => {
  const out = process.env.BOLO_WEBRTC_MEASUREMENT
  if (!out || !runs.length) return
  const doc = {
    date: new Date().toISOString().slice(0, 10),
    what:
      'A voice-agent turn over WebRTC in headless Chromium (Playwright) against a real bolo serve: fake microphone ' +
      '-> RTCPeerConnection (Opus) -> bolo (silero-vad, recogniser, language model, voice) -> remote audio track. ' +
      'Times are from the Start click; "agent_sound" is the first inbound audio with energy above silence (getStats ' +
      'totalAudioEnergy). The fake microphone loops the sample with 8 s of silence after it, starting when the ' +
      'microphone opens, so the first words may be lost before the call connects and the transcript time includes ' +
      'waiting for the sample to come round again.',
    browser: browserVersion,
    chromium_flags: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--use-file-for-fake-audio-capture=<en-short.wav + 8 s of silence>',
    ],
    microphone: { sample: path.basename(SAMPLE), text: SAMPLE_TEXT, silence_after_s: 8 },
    models: { ...MODELS, llm: 'fake-llm (server/tests/fake_llm.py: deterministic, echoes the transcript)' },
    network: 'browser and server on the same machine; no STUN or TURN',
    runs,
    not_covered:
      'headed Chrome, Edge, Firefox and Safari; a real microphone and speakers; NAT traversal, TURN, calls across ' +
      'the internet',
  }
  fs.writeFileSync(out, JSON.stringify(doc, null, 1) + '\n')
})
