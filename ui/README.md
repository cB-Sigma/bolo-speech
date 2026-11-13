# bolo console

The web console for [bolo](../README.md): a playground for speech to text and text to speech, a
model catalog, a live latency dashboard, a concurrency benchmark and copy-paste client snippets.

It is a static single-page app (React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts). The
production bundle is written to `../server/src/bolo/_ui/` and served by the bolo server at `/`, so
installing bolo with pip never needs Node.

## Requirements

Node 22 or newer and npm.

```bash
npm install
```

## Develop

Against a running bolo server (default `http://127.0.0.1:8000`, override with `BOLO_SERVER`):

```bash
bolo serve                       # in another terminal
npm run dev                      # http://127.0.0.1:5173, API calls are proxied to the server
BOLO_SERVER=http://10.0.0.5:8000 npm run dev
```

Without a server, against the built-in mock:

```bash
npm run dev:mock
```

The mock (`src/mock/`) implements the same HTTP, WebSocket and server-sent-event contract in the
page: a catalog, pulls with progress, simulated traffic with a load curve, streaming recognition
driven by a real voice-activity detector on whatever audio you send, streaming synthesis, and
optional components (it starts without eSpeak NG, so two voices show the install flow). The
header shows a **Mock data** badge so its numbers are never mistaken for measurements. Flags for
edge cases live in local storage:

```js
localStorage.setItem('bolo.mock', JSON.stringify({ auth: true })) // require the key "bolo-demo"
localStorage.setItem('bolo.mock', JSON.stringify({ offline: true })) // server unreachable
```

The application code is identical in both modes; only the transport in `src/lib/api/config.ts` is
swapped.

## Scripts

| Script                                | What it does                                                                     |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| `npm run dev`                         | Vite dev server, proxying `/api`, `/v1`, `/healthz`, `/metrics` to `BOLO_SERVER` |
| `npm run dev:mock`                    | Vite dev server with the in-page mock API                                        |
| `npm run build`                       | Type-check, then write the production bundle to `../server/src/bolo/_ui/`        |
| `npm run build:mock` / `preview:mock` | Build with the mock into `dist-mock/` and serve it on port 4173                  |
| `npm run typecheck`                   | `tsc` in strict mode, no output                                                  |
| `npm run lint`                        | ESLint and a Prettier check                                                      |
| `npm run format`                      | Prettier, in place                                                               |
| `npm test`                            | Unit and component tests (Vitest, Testing Library)                               |
| `npm run e2e`                         | End-to-end and accessibility tests (Playwright, axe) against the mock build      |

## Build

```bash
npm run build
```

The output directory is emptied first. Asset URLs are relative and routing uses the URL hash, so the
same bundle works at `/` and behind a reverse proxy under any sub-path. The API base is the
directory the page was loaded from; set `VITE_BOLO_SERVER` at build time, or the `bolo.server` local
storage key at run time, to point the console at a different origin.

## Live demo (GitHub Pages)

`npm run build:demo` produces a static site with no server behind it, published by
`.github/workflows/pages.yml` to <https://cb-sigma.github.io/bolo/>. In this build:

- Speech to text really runs, in the page, with the in-browser recogniser (microphone, file upload
  and two bundled samples). Nothing is simulated and no audio leaves the device.
- The Latency screen shows only the traces of what was run in the tab during the visit (model load,
  endpointing, queueing, decode). The Benchmark screen runs a real sweep in the page and says plainly
  that it measures a single in-page recogniser, not server concurrency.
- The Models screen lists the browser model (download and cache state, size, licence) and a
  read-only snapshot of a server's catalog, `src/demo/catalog-snapshot.json`, labelled "available
  when you run the server". Refresh it with `node scripts/export-catalog.mjs --base <server>`.
- Everything that needs a server (text to speech, pull/load/unload, the phone-line simulation,
  compare mode, Wyoming) is disabled with a one-line reason and the install command.
- "Share this result" puts the session's latency summary in the URL hash. There is no backend; a
  shared link just shows those numbers, labelled as someone else's measurement.

GitHub Pages cannot send the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers
that browsers require for WebAssembly threads, so the demo registers a small service worker
(`src/demo/sw.js`) that adds them to every response and reloads the page once on the first visit.
The same worker precaches the bundle, so the demo works offline afterwards (the model itself is
cached by the recogniser). Where service workers are unavailable the demo still runs, on one thread.

```bash
npm run build:demo
npm run serve:demo            # http://127.0.0.1:4174/bolo/
BOLO_E2E_MODEL_DIR=/path/to/mirror npm run e2e:demo
```

The demo is a build flag (`--mode demo`); the bundle served by bolo is not affected by it.

## Tests

```bash
npm test          # latency maths, audio utilities, API client, mock server, components
npm run e2e       # every screen in dark and light at 390 px and 1440 px, flows, axe
```

The end-to-end tests start `vite preview` on the mock build themselves. They use the Playwright
version pinned in `package.json`; install its Chromium once with `npx playwright install chromium`
if you do not already have it. Chromium is launched with a fake microphone that plays
`src/mock/assets/en-ask-not.wav`, so the capture path (getUserMedia, AudioWorklet, resampling,
WebSocket) is exercised for real.

The in-browser recogniser test downloads nothing by itself. Give it a local mirror of the model to
run it (see the header of `e2e/browser-stt.spec.ts`):

```bash
BOLO_E2E_MODEL_DIR=/path/to/mirror npm run e2e -- browser-stt
```

## Layout

```
src/
  lib/api/          typed client for the native API, SSE reader, transport and URL handling
  lib/latency/      percentiles, histograms, budget, network/model split, timeline layout, WER
  lib/audio/        resampler, G.711 μ-law/A-law, PCM framing, WAV, capture worklet, gapless player
  lib/stt/          transcript and latency tracking, server and in-browser transcribers, energy VAD
  lib/tts/          streaming synthesis session with per-chunk timing and underrun detection
  lib/browser-stt/  web worker running Moonshine with ONNX Runtime Web (WebGPU or WebAssembly)
  components/ui/    shadcn/ui components (generated; see components.json)
  components/       charts, panels, layout, screen-specific pieces
  routes/           one file per screen, each its own chunk
  mock/             in-page implementation of the server contract (development and tests)
  demo/             live-demo backend fed by real in-browser work, service worker, catalog snapshot
e2e/                Playwright specs for the normal build
e2e-demo/           Playwright specs for the live demo build
```

## In-browser speech recognition

The playground can run a small English model in the page, with no server round trip, and compare it
with the server on the same audio. The runtime (`@huggingface/transformers` with ONNX Runtime Web)
lives in a web worker that is only fetched when the mode is first used. It prefers WebGPU and falls
back to WebAssembly with SIMD; WebAssembly threads need a cross-origin-isolated page. The model
weights and the ONNX Runtime binary are downloaded on first use and cached by the browser. Details,
licences, the headers the server should send and how to self-host the files are in
[`../docs/ui-server-notes.md`](../docs/ui-server-notes.md).

## Design notes

- One accent colour; status colours (good, warning, critical) are reserved for state and always
  paired with an icon or words.
- Chart colours are a fixed, ordered categorical palette checked for colour-vision deficiencies in
  both themes. Every chart has hover inspection and a table view.
- Numbers use a tabular monospace. Motion is limited to 150–250 ms transitions and is disabled under
  `prefers-reduced-motion`.
- Inter and JetBrains Mono are bundled (SIL Open Font License 1.1) so the console renders the same
  offline.

## Licence

Apache-2.0, like the rest of bolo. Runtime dependencies are MIT, Apache-2.0, BSD or ISC licensed;
the bundled fonts are OFL-1.1. Two things in `node_modules` are not permissive and are not shipped:
`@axe-core/playwright` (MPL-2.0), used only by the test suite, and the prebuilt libvips binaries
(`@img/sharp-libvips-*`, LGPL-3.0-or-later) that `sharp` installs as a Node-only dependency of
`@huggingface/transformers`. The browser build of that library does not use `sharp`, so neither ends
up in the bundle.
