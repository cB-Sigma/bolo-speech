# Console ↔ server notes

Working notes on what the web console (`ui/`) needs from the server that serves it, and what was
found while running the console against a real `bolo serve`.

## Serving the bundle

- The bundle in `server/src/bolo/_ui/` is fully static: `index.html`, `favicon.svg`, `og.png` and
  hashed files under `assets/`. Asset URLs are relative and the app uses hash routing
  (`/#/latency`), so only `/` (and `/index.html`) has to return `index.html`. No SPA fallback route
  is needed, and the console works unchanged behind a reverse proxy under a sub-path.
- The console calls the API relative to the directory it was loaded from (`./api/v1/...`,
  `./v1/...`, `./healthz`), so the API must be mounted next to it.
- Caching: `assets/*` are content-hashed and can be served with
  `Cache-Control: public, max-age=31536000, immutable`; `index.html` should be `no-cache`.
- `index.html` contains one small inline script (it applies the saved theme before first paint). If
  a Content-Security-Policy is added, allow it by hash or nonce.

## Headers for the in-browser recogniser

The playground can run speech recognition in the page with ONNX Runtime Web. It works without any
special headers, on one WebAssembly thread. To let it use several threads the page must be
cross-origin isolated, which needs these two response headers on `index.html` and on everything
under `assets/` (the worker script is loaded from there):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

`credentialless` (rather than `require-corp`) keeps the cross-origin downloads below working
without those hosts having to send `Cross-Origin-Resource-Policy`. Safari does not support
`credentialless`; there the console simply stays single-threaded. The Vite dev and preview servers
in `ui/` send both headers, and so does `bolo serve` (checked: `same-origin` / `credentialless`), so
against the real server the in-browser mode runs multi-threaded. Hosts that cannot set headers, such
as GitHub Pages, are covered by the live demo's service worker (see the end of this file).

If a Content-Security-Policy is used, the in-browser mode needs:

```
script-src  'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co https://cdn.jsdelivr.net
worker-src  'self' blob:
```

## What the in-browser mode downloads

In the console served by the server, nothing is downloaded until the user switches the playground
to "In browser" or "Compare" and starts a take, presses down on the microphone button, or presses
the model status. The live demo fetches the selected model while the page is idle, unless the
browser reports Save-Data or a 2G connection, in which case it waits to be asked. Files are cached
by the browser (Cache Storage, `transformers-cache`) and can be removed from the same panel.

| What | Source | Size | Licence |
| --- | --- | --- | --- |
| Moonshine Tiny, English, 8-bit quantised ONNX (`encoder_model_quantized.onnx`, `decoder_model_merged_quantized.onnx`, tokenizer and config JSON). The default. | <https://huggingface.co/onnx-community/moonshine-tiny-ONNX> (conversion of <https://huggingface.co/UsefulSensors/moonshine-tiny>) | 32.1 MB | MIT |
| Whisper Tiny Hindi, 8-bit quantised ONNX. Hindi only. | <https://huggingface.co/astronova001/whisper-tiny-hindi-ONNX>, a community conversion of <https://huggingface.co/ARTPARK-IISc/whisper-tiny-vaani-hindi>. The conversion repository carries no licence tag of its own; the weights it converts are Apache-2.0. | 124 MB | Apache-2.0 |
| Whisper Tiny, multilingual (99 languages), 8-bit quantised ONNX | <https://huggingface.co/onnx-community/whisper-tiny> (conversion of <https://huggingface.co/openai/whisper-tiny>, Apache-2.0; the conversion repository has no licence tag) | 43.6 MB | Apache-2.0 |
| Whisper Base, multilingual (99 languages), 8-bit quantised ONNX | <https://huggingface.co/onnx-community/whisper-base> | 79.7 MB | Apache-2.0 |
| ONNX Runtime Web WebAssembly binary and loader (`ort-wasm-simd-threaded.wasm` / `.mjs`; the `.asyncify` build when WebGPU is used) | jsDelivr, `onnxruntime-web` at the exact version `@huggingface/transformers` depends on | about 14 MB (27 MB for the WebGPU build) | MIT |

Sizes are the sums of the files listed on the Hub on 2026-11-13. Whisper Small was left out: its
8-bit files come to 249 MB, too much for a phone, and it was not measured.

### Indian languages in the browser: what was measured

The language select lists Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam,
Punjabi, Urdu, Nepali and Assamese (the ones in Whisper's language list; Odia is not) with their own
names. Choosing one while an English-only model is selected opens a dialog that names the models
that can do it and their download size.

Accuracy was measured on Hindi only (13 recordings: 12 real speakers from the FLEURS dev split and
the built-in sample; full record in `docs/measurements/browser-whisper-hindi.json`):

| Model | Word error rate | Character error rate | Script of the output |
| --- | --- | --- | --- |
| Whisper Tiny Hindi | 40 % | 27 % | Devanagari |
| Whisper Base (multilingual) | 102 % | 110 % | mostly Latin, some Urdu |
| Whisper Tiny (multilingual) | 231 % | 264 % | mostly Latin |

So the multilingual Tiny and Base models are not usable for Hindi, and the console says so next to
the model; "Whisper Tiny Hindi" is the suggested choice and is good enough to show Hindi working,
not for real use. No fine-tuned model in a browser-ready ONNX layout with a clear permissive licence
was found for the other languages, so they fall back to multilingual Whisper with a plain warning.
All Whisper models decode a fixed 30-second window per pass; in headless Chromium on the (busy)
4-core build machine they ran slower than real time, so partial results are skipped for them and
text arrives seconds late. The server's models for Indian languages are much better and faster.

### Runtime and model hosts under cross-origin isolation

Checked with `curl` on 2026-11-13, sending `Origin: https://cb-sigma.github.io`:

- jsDelivr (`ort-wasm-simd-threaded.wasm`, `.mjs`, and the `.asyncify` build) answers 200 with
  `access-control-allow-origin: *` and `cross-origin-resource-policy: cross-origin`.
- `huggingface.co/<repo>/resolve/main/...` answers 302 (large files) or 307 (small files) with
  `access-control-allow-origin` set to the requesting origin; the CDN it redirects to answers 200
  with `access-control-allow-origin: *`.

Both are fetched with CORS, which is what `Cross-Origin-Embedder-Policy: require-corp` (the mode the
demo's service worker uses outside Chromium) requires. The runtime files are therefore still loaded
from jsDelivr rather than copied into the Pages artifact. The first download of the runtime binary
is now made by the worker itself into the same cache entry the library uses, so that progress can be
shown; the library then finds it there, as on any later visit.

The JavaScript side (`@huggingface/transformers`, Apache-2.0) is part of the console bundle as a
lazily loaded worker chunk (about 550 kB before compression). The ~27 MB WebAssembly binary is
deliberately **not** copied into the bundle.

Endpointing in the browser uses a small energy-based voice activity detector written for the
console (`ui/src/lib/stt/vad.ts`), so no second model is fetched.

For air-gapped installs both sources can be redirected without rebuilding, by setting a local
storage key in the browser:

```js
localStorage.setItem('bolo.browserStt', JSON.stringify({
  modelHost: 'https://models.internal.example',        // serves /onnx-community/moonshine-tiny-ONNX/resolve/main/...
  ortWasmBase: 'https://static.internal.example/ort/', // serves ort-wasm-simd-threaded*.{wasm,mjs}
}))
```

A natural follow-up would be for the server to offer both under its own origin (for example
`/api/v1/browser-assets/...`) and advertise that in `/api/v1/info`; the console would then prefer it.

`stallSeconds` in the same object changes how long a load may go without progress before the
console reports it as stuck (default 40).

## Microphone, first use and diagnostics

- The microphone is requested in the same turn as the click or tap (`getUserMedia` first, then the
  `AudioContext`, both before anything is awaited), in parallel with connecting or loading the
  in-browser model. Audio captured in the meantime is buffered (up to 60 s) and handed over when
  the recogniser is ready. Latency for that first stretch is measured from the hand-over.
- A refused or failed microphone opens a help panel in place of a one-line message. It names the
  likely cause for this browser (an in-app browser, a saved block for the site, the operating
  system, a dismissed prompt, no device, a busy device, an insecure page) with steps for the
  detected platform. In-app browsers are detected conservatively from the user agent and the
  attempt is never blocked on that detection.
- Loading shows the stage (runtime, model, compile, warm-up) with bytes. If nothing moves for
  `stallSeconds` the console says which stage is stuck and offers "Retry single-threaded" (one
  WebAssembly thread, no WebGPU).
- `?debug=1` (before or after the `#`) adds a diagnostics panel to every screen: user agent, secure
  context, cross-origin isolation, SharedArrayBuffer, the COEP mode, service-worker state, the
  microphone permission, every model and runtime request with status, size and time, and errors.
  It has a Copy button. `?debug=0` turns it off again.

## API usage

The console uses only: `GET /api/v1/info`, `/catalog`, `/stats`, `/requests`, `/samples`,
`/samples/{file}`, `/jobs/{id}`, `/events` (SSE), `POST /api/v1/models/pull`,
`/models/{id}/load|unload`, `DELETE /api/v1/models/{id}`, `POST /api/v1/bench`,
`WS /api/v1/stt/stream`, `WS /api/v1/tts/stream`, `POST /v1/audio/speech` (the "Chunked HTTP"
transport in the TTS playground, `response_format=wav`), and `GET /healthz` (round-trip timing).

- **Auth.** HTTP requests and the event stream send `Authorization: Bearer <key>` (the event
  stream is read with `fetch`, not `EventSource`, for that reason). WebSockets pass `?api_key=<key>`.
  A 401/403 on the event stream, or `auth_required: true` with no stored key, opens the key dialog.
- **Latency split.** The playground shows "speech end → final" as measured in the browser and
  splits it into model time (`latency_ms` on the final), network (median round trip of
  `{"type":"ping"}` → `pong` on the open stream, falling back to `GET /healthz`), and the remainder
  (endpointing silence and queueing).
- **TTS stream.** The console sends `text`, then `flush`, then `stop`, and treats `done` as the end
  of audio. Chunks with empty `text` are folded into the preceding sentence in the timeline.

## Found while integrating against `bolo serve --fake-engine`

Checked with the sherpa-onnx engine and the models installed on the build machine (4 CPU cores, no GPU). None of these
block the console; they are listed so they can be fixed or documented on the server side.

1. **Some TTS models returned no audio and no error** (`kitten-nano-0.8-int8`,
   `piper-en-us-amy-low`). Resolved by the components work below: the cause was a missing eSpeak NG,
   and the server now answers such requests with `component_missing`.
2. **8 kHz telephony audio that is properly band-limited gets no speech detected.** With
   `encoding=mulaw&sample_rate=8000` on the STT WebSocket (`moonshine-tiny-en`, VAD `silero-vad`),
   audio produced by a plain `ffmpeg -ar 8000 -f mulaw` is segmented and transcribed, but the same
   recording low-passed below 4 kHz first (as a real phone line does, and as the console's
   "Phone line" simulation does) produced no `vad` events and no finals: only `metrics` with
   `rtf: 0` and `done`. Reproduce without the console:
   `ffmpeg -i speech.wav -af "lowpass=f=3400:p=2,lowpass=f=3400:p=2,lowpass=f=3400:p=2" -ar 8000 -f mulaw lp.ulaw`
   and stream `lp.ulaw` in 160-byte frames. The same band-limited audio is transcribed correctly
   through `POST /v1/audio/transcriptions`, so it looks like the voice-activity step on the 8 kHz
   streaming path rather than the recogniser. Until this is fixed the playground's phone-line toggle
   shows an empty transcript against the real server (it works against the mock).
3. **`flush` does not end a TTS stream.** `done` is only sent after `stop`. The contract's message
   list reads as if `flush` might be enough; the console sends both.
4. **`partial` can arrive before `vad: speech_start`** for the same utterance (seen with
   `fake-stt-streaming`). The console handles either order.
5. **A trailing final after `stop`.** `fake-stt-streaming` emits one more one-word final with
   `t0 == t1` at the end of the stream. Harmless, but it shows up as an extra utterance.
6. **Realtime traces repeat phase names** (several `vad`/`decode` entries, one per utterance) and
   their `total_ms` is the length of the whole connection. The waterfall draws them as they come and
   caps its axis so that one long session does not flatten every other row.
7. **Everything is `validated: "untested"` and `measured: null`** at the time of testing, so the
   "Validated on CPU" mark and the "Measured by the maintainers" block in the model drawer could
   only be exercised with the mock. (A later server build reports 70 entries as validated with
   measurements; the console shows them, and the demo's catalog snapshot was exported from that build.)

Things that matched the contract and its `CHANGED:` notes without adjustment: `info`, `catalog`
(including the extra `license_url`, `commercial_use`, `measured`, `custom` fields), `stats`,
`requests`, the flat SSE event shapes, bench jobs with partial `result.points`, the STT WebSocket
(`ready` with `mode`, `vad`, `partial` with `stable_chars`, `final` with `words`, `metrics`,
`pong`), model pull with progress, load, unload and delete, the built-in speech samples, and error
bodies with `detail`.

## Optional components (eSpeak NG)

The console uses `GET /api/v1/components`, `POST /api/v1/components/{id}/install`
(`{"accept_license": true}`), the `requires` / `available` fields on catalog entries, the
`component` server-sent event, and the `component_missing` error code.

- Models: a "Needs eSpeak NG" mark on entries with `available: false`, a "No extra components"
  filter (hides anything with a non-empty `requires`), and an "Optional components" card. Load is
  disabled for unavailable entries.
- Text to speech: an inline notice for an unavailable voice, with an install button that opens a
  confirmation (licence name and link, what will be installed), follows the job by polling
  `/api/v1/jobs/{id}`, then refreshes the catalog so the voice is usable without a reload. When
  `installable` is false the notice shows `install_blocked_reason` and the copyable
  `manual_command` instead. A `component_missing` error from the TTS WebSocket or from
  `POST /v1/audio/speech` leads to the same notice.
- Connect: a status row per component. A server without the route (404) is treated as having none.

Checked against the real server on 2026-11-13, run from a scratch Python environment that could
see bolo but not `espeakng-loader`: status and counts (82 voices need it), the mark and filter, the
notice, the install (about one second from a warm package cache, into that scratch environment),
the `component` event, and synthesis with `piper-en-us-amy-low` straight afterwards (first audio in
about 120 ms). Observations:

- When the component is already installed the server reports `installable: false` with a null
  reason. The console only looks at `installable` for components that are not installed, so this is
  harmless, but `true`/`null` semantics for the installed case could be documented.
- `POST /api/v1/models/{id}/load` succeeds for a model whose component is missing; only synthesis
  fails. The console disables Load for such models, which is stricter than the server.
- The HTTP status for `component_missing` on `/v1/audio/speech` is 424; the console keys on the
  code, not the status.
- The install-refused path (`installable: false` on a missing component) was exercised with the
  mock only.

## The live demo build (no server)

`ui` can also be built as a static site for GitHub Pages (`npm run build:demo`, published by
`.github/workflows/pages.yml`). It needs nothing from the server at run time, but two things tie it
to the server at build time:

- **Catalog snapshot.** `ui/src/demo/catalog-snapshot.json` is exported from a running server with
  `node ui/scripts/export-catalog.mjs --base http://127.0.0.1:8000` (fake-engine and custom entries
  are left out, machine state is stripped). The demo shows it read-only, as "available when you run
  the server". Re-export it when the catalog changes materially.
- **Install command.** The demo tells visitors how to get the real thing using the command in
  `ui/src/lib/api/config.ts` (`INSTALL_COMMAND`). Keep it in step with the README.

In the demo, optional components are listed read-only (how many catalog voices need eSpeak NG, its
licence) with a note that installing them is a server feature.

Cross-origin isolation on a host that cannot send headers is provided by `ui/src/demo/sw.js`, a
small service worker that adds COOP/COEP to every response it relays (the approach popularised by
coi-serviceworker, MIT; this is an independent implementation) and caches the bundle for offline
use. It asks for `credentialless` on Chromium and `require-corp` elsewhere.

In-browser text to speech was considered for the demo and left out. Checked on 2026-11-13:
Kokoro (`onnx-community/Kokoro-82M-v1.0-ONNX`, Apache-2.0, 86–92 MB quantised) and the Piper/Kitten
voices need a phonemiser, and the web builds available (`kokoro-js` → `phonemizer`) bundle eSpeak
NG, which is GPL-3.0; SpeechT5 (`Xenova/speecht5_tts` plus vocoder, MIT) needs about 177 MB at 8-bit
quantisation; Supertonic is under an OpenRAIL licence and about 260 MB. None is both permissively
licensed end to end and small enough to download into a visitor's tab.
