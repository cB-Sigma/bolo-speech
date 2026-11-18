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

## Server flags that replaced console heuristics (2026-11-14)

Six things the console used to work out for itself are now reported by the server. The console
reads the flag when it is there and keeps the old reasoning only for older servers.

| What | Before (in the console) | Now (from the server) |
| --- | --- | --- |
| Can a Realtime pipeline be used? | `speech-to-speech`: a URL is set, or a model is named and the llama-server component is installed | `capabilities.realtime.pipelines.<id>.available` and `.configured`, with `.reason` for speech-to-speech |
| Can a catalog model run at all? | the description contains "Not runnable yet" | `runnable: false` and `not_runnable_reason` on the entry |
| Models whose files are not published | absent from `GET /api/v1/catalog`, so the console only met them by id | listed, with `status: "pending"` |
| Which pipeline answered a voice-agent turn | not on the trace | `pipeline` on `kind: "agent"` traces (shown in the request drawer) |
| Is a model that is not installed downloaded on first use? | unknown: the console always said "pull it first" | `info.auto_pull` (the playgrounds word their hint from it) |
| Does the server have word lists? | probe `GET /api/v1/wordlists` and read a 404 as "no" | `info.capabilities.wordlists` (the probe is skipped when the server announces capabilities without it) |

Cloud providers added three screens' worth of API use: `GET/PUT/DELETE /api/v1/providers[/{name}]`
and `POST …/test` (Providers), `cloud` / `notice` / `provider` / `fallback` on catalog entries
(Models: the "Cloud" mark and the "Runs" filter), and for Compare the response headers
`x-request-id`, `x-bolo-model`, `x-bolo-fallback` and `x-bolo-estimated-cost` plus
`GET /api/v1/requests/{id}` for the server's own timings and the estimated cost. With CORS (a
console on another origin) those headers are readable because the server lists them in
`Access-Control-Expose-Headers`.

The key field of the Providers screen is write-only by construction: the form state starts with an
empty key every time, the server never sends one, and saving without typing sends no `api_key`
(the server keeps the stored key).

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

## Speech synthesis in the browser

The text-to-speech screen of the live demo runs voices in the page; the console served by
`bolo serve` has the same screen as an "In browser" mode next to "Server" (there is no "Compare"
mode for speech synthesis). Nothing in it needs the server.

### How it works

- A worker (`ui/src/lib/browser-tts/worker.ts`) runs the voice's VITS graph directly with ONNX
  Runtime Web (`onnxruntime-web/wasm`, the version `@huggingface/transformers` uses, so the
  WebAssembly binary in Cache Storage is shared with the recogniser and fetched once), on up to 4
  threads when the page is cross-origin isolated.
- **Character voices** (IISc SYSPIN, Coqui VITS) need no phonemiser. The front end
  (`ui/src/lib/browser-tts/text.ts`) does what their original Coqui front end did: NFC, lower case,
  Coqui's `multilingual_cleaners`, one id per character from the voice's table, unknown characters
  skipped, a blank id between every pair and at both ends. After each model call the audio goes
  through the same pause shortening sherpa-onnx applies (`ScaleSilence`, scale 0.2). Checked
  against sherpa-onnx 2.0.1 with the noise switched off: for 7 test sentences across Hindi,
  Maithili, Bhojpuri, Marathi, Kannada and English the waveforms are identical (same length,
  largest difference below 1e-3); the ids are kept as unit-test fixtures
  (`fixtures/sherpa-tokens.json`, made by `ui/scripts/voices/fixtures.py`). sherpa-onnx does not
  apply Coqui's cleaners, so for text with capitals, `;`, `:`, `-`, brackets or double quotes the
  browser follows the training text and the server does not; and sherpa-onnx does not NFC its
  input, so a precomposed nukta letter (क़, U+0958) maps to a different id there.
- Numbers are spelt out before synthesis (`numbers.ts`: English, Hindi, Marathi and Kannada names,
  Indian grouping, decimals, per cent; Maithili and Bhojpuri use the Hindi names). Characters a
  voice has no symbol for are listed under the text box instead of vanishing silently; a missing
  danda (the Bhojpuri voices have none) becomes a full stop.
- **Piper voice** (US English "amy", low): phonemes come from uccharan (Apache-2.0, vendored in
  `ui/src/vendor/uccharan`, WebAssembly module 1.7 MiB plus 0.7 MB of English data), `piper`
  profile, ids `^ _ p _ … $`. No eSpeak NG build is used anywhere.
- Text is split into sentences (a long first sentence at its first comma); each one is
  synthesised and scheduled on the Web Audio clock as soon as it is ready, so the first plays while
  the rest are made. Time to first audio is measured from the moment the voice is ready; the
  one-time download and compile are shown separately. The real-time factor is model time per second
  of audio.
- The round trip feeds each synthesised sentence to the in-browser recogniser: Whisper Tiny Hindi
  for the Hindi voice, Moonshine Tiny for the two English voices. There is none for Maithili,
  Bhojpuri, Marathi or Kannada; the button says so.

### Memory on iPhone and iPad

ONNX Runtime Web 1.19 and later ships only builds whose loader creates a *shared*
`WebAssembly.Memory` with a 4 GiB maximum, even on one thread (checked in 1.31: the plain,
asyncify, jsep and jspi builds all do; there is no non-threaded artefact any more, so "load the
non-threaded file" is not an option). Engines reserve address space for the whole maximum of a
shared memory up front. WebKit on iOS has little of it: one such memory (the recogniser's) fits,
a second one in the same page fails with `RangeError: Out of memory`, which ONNX Runtime reports as
`no available backend found. ERR: [wasm] RangeError: Out of memory`, on any number of threads. That
is what an iPhone (Edge, i.e. WebKit) showed for the Piper voice after the demo had loaded the
recogniser. The cause is inferred, not reproduced on WebKit (Playwright's WebKit is not installed
on the build machine; V8 shows the same pattern: a 4 GiB shared memory needs about 10 GB of address
space). What the console does now:

- The synthesiser loads the runtime's JavaScript with the maximum lowered to 1 GiB (the
  WebAssembly module accepts any maximum up to its own; a 13-second sentence of a 114 MB voice
  needs about 0.3 GiB; amy uses 144 MiB after loading). If the loader does not look as expected
  the cap is skipped and the diagnostics say so.
- On WebKit (every browser on iPhone and iPad, and Safari) only one model runtime is kept alive:
  loading the voice releases the recogniser's worker and the other way round; each loads again by
  itself when next needed (from the browser cache).
- Any out-of-memory error while loading (recogniser or synthesiser) releases the other runtime,
  switches to one-at-a-time for the rest of the visit and retries once on one thread before
  showing anything. If that fails too, the panel says to close other tabs.
- `?debug=1` shows the runtime's memory cap and size after loading and as it grows, the page's
  memory where the browser reports it, the device memory class, and whether runtimes are kept one
  at a time.

The e2e suite simulates the WebKit limit by refusing large shared memories in the synthesiser's
worker, and runs the one-at-a-time mode with an iPhone user agent.

### Where the voices come from

GitHub release assets are served without CORS headers, so a page cannot fetch them. The voices are
therefore published **with the Pages site, same origin, under `voices/`** and are not in git.
`.github/workflows/pages.yml` runs `ui/scripts/voices/build.py`, which

1. downloads the SYSPIN archives that the "Voices" workflow (`tools/voices/`) published as assets of
   the `voices-v1` release, and the Piper voice from `rhasspy/piper-voices` at a pinned revision
   (SHA-256 checked);
2. shrinks each graph (`shrink.py`): convolution weights outside the text encoder, duration
   predictor and (for Piper) decoder become 8-bit integers with one scale per output channel,
   everything else 16-bit floats; ONNX Runtime expands them to fp32 constants when the session is
   created (`session.disable_quant_qdq=1`, without which the DequantizeLinear nodes ran on every
   call and made synthesis about 10 % slower), so computation stays fp32;
3. checks each shrunk voice against the original with the noise off (same length, mean log-mel
   difference on the loudest frames below 0.08; measured 0.017 to 0.023) and writes
   `voices/index.json` with sizes and SHA-256 of every file.

It needs only `numpy`, `onnx` and `onnxruntime` and takes a few minutes; the output is cached by
the workflow (key: the release's `voices.json` and the build scripts). Voices whose archive is not
published yet are left out (`--skip-missing`) and the demo says "not published with this build".
**Order of operations:** run the "Voices" workflow once (it creates `voices-v1`),
then the "Live demo" workflow (it also runs by itself when "Voices" completes). The Pages artifact
grows by about 320 MB; the largest single file is 36.5 MB.

The console served by `bolo serve` fetches the voices from the public demo
(`https://cb-sigma.github.io/bolo-speech/voices/`; GitHub Pages sends `Access-Control-Allow-Origin: *`).
To use a copy of your own: `localStorage.setItem('bolo.browserTts', JSON.stringify({ voicesBase:
'https://static.internal.example/voices/' }))`. A natural follow-up is for the server to offer
the voices it has installed under its own origin. With a Content-Security-Policy, add the voices
host to `connect-src`.

| Voice | Language | Download | Weights / data licence | Made by |
| --- | --- | --- | --- | --- |
| `syspin-hi-female` | हिन्दी Hindi | 36.5 MB | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `syspin-mai-female`, `syspin-mai-male` | मैथिली Maithili | 36.5 MB each | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `syspin-bho-female`, `syspin-bho-male` | भोजपुरी Bhojpuri | 36.5 MB each | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `syspin-mr-female` | मराठी Marathi | 36.5 MB | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `syspin-kn-female` | ಕನ್ನಡ Kannada | 36.5 MB | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `syspin-en-female` | English (India) | 36.5 MB | MIT / CC-BY-4.0 | SYSPIN, SPIRE Lab, IISc |
| `piper-en-us-amy-low` | English (US) | 25.7 MB (25.0 MB voice + 0.7 MB uccharan data) | MIT (piper-voices) / Amy dataset, mimic3-voices (licence not stated per voice) | Rhasspy / Piper |

Original fp32 sizes: 114 MB per SYSPIN voice, 63 MB for amy. Plus, once, the ONNX Runtime binary
from jsDelivr (about 12 MB, shared with the recogniser).

### Quality of the shrunk voices

Round trip, original vs shrunk, three sentences per voice with the normal sampling noise,
transcribed by sherpa-onnx (IndicConformer per language; Hindi model for Bhojpuri; Whisper Base for
English); character error rate:

| Voice | fp32 | shrunk |
| --- | --- | --- |
| Hindi female | 9.2 % | 6.2 % |
| Maithili female / male | 6.2 % / 3.7 % | 6.2 % / 1.2 % |
| Bhojpuri female / male | 15.1 % / 6.8 % | 12.3 % / 13.7 % |
| Marathi female | 0.0 % | 2.6 % |
| Kannada female | 0.0 % | 0.0 % |
| Indian English female | 3.7 % | 5.6 % |

The differences go both ways and are within the run-to-run variation the sampling noise causes.
An 8-bit text encoder, by contrast, changed the output clearly (log-mel difference 0.64), which is
why it stays 16-bit. Nobody who speaks these languages has listened to the output.

### Measured in headless Chromium

Measured on 2026-11-13 in headless Chromium on the build machine (4 cores of an Intel Xeon @ 2.10 GHz,
no GPU, shared with other jobs: load average 4 to 7), voices served same-origin as on Pages, 4
WebAssembly threads. Full record: `docs/measurements/browser-tts.json`. "First use" is a fresh
profile (download from a local server, so network time is close to zero; add your download time),
"from cache" is after a reload. Text: the voice's sample plus one more sentence (4 to 8 s of audio).

| Voice | Download | Ready, first use | Ready, from cache | Time to first audio | Real-time factor |
| --- | --- | --- | --- | --- | --- |
| `syspin-hi-female` | 36.5 MB | 3.8 s | 1.7 s | 1843–3630 ms | 1.93–3.02 |
| `syspin-mai-female` | 36.5 MB | 4.2 s | 2.2 s | 915–1827 ms | 1.83–2.89 |
| `syspin-mai-male` | 36.5 MB | 3.6 s | 1.8 s | 1375–2624 ms | 1.85–2.13 |
| `syspin-bho-female` | 36.5 MB | 6.1 s | 1.6 s | 1102–2765 ms | 1.63–2.73 |
| `syspin-bho-male` | 36.5 MB | 3.6 s | 2.0 s | 961–1314 ms | 1.59–1.77 |
| `syspin-mr-female` | 36.5 MB | 3.7 s | 1.9 s | 1583–2017 ms | 1.67–1.78 |
| `syspin-kn-female` | 36.5 MB | 4.4 s | 1.7 s | 2031–2822 ms | 1.54–2.47 |
| `syspin-en-female` | 36.5 MB | 4.4 s | 2.2 s | 1909–2670 ms | 1.59–1.73 |
| `piper-en-us-amy-low` | 27.1 MB | 4.0 s | 2.1 s | 230–292 ms | 0.32–0.38 |

So on this machine the SYSPIN voices run **slower than real time** (real-time factor 1.5 to 3):
the first sentence arrives after 1 to 3 s and later sentences make playback wait (the underruns
the screen shows are real). The Piper voice (16 kHz, smaller decoder) runs at 0.2 to 0.4 with first
audio in about 0.2 to 0.3 s. In Node, single-threaded, ONNX Runtime Web ran the Hindi voice at a
real-time factor of 2.3 where native ONNX Runtime took 0.65 on the same busy machine: WebAssembly
costs about 3.5× here. With isolation switched off (1 thread) the numbers were about the same as
with 4 threads, so on this loaded machine extra threads did not help; on an idle machine they
should. A recent laptop or phone will do better than this shared server; it was not measured.

## Small audio models in the page

The live demo runs the server's newer audio features in the page with small models of their own (code in
`ui/src/lib/browser-audio/`, one worker per task, registered with the downloads list, the Loaded models view,
the keep-alive and the one-runtime-at-a-time rule of phones). Each keeps the id and licence of the server's
catalog entry. Files come from Hugging Face at the server's pinned revisions, except where noted.

| Feature (page) | Models (catalog id) | Source | Size | Licence |
| --- | --- | --- | --- | --- |
| Denoise first (Speech to text, live mic and files) | `gtcrn-simple` | sherpa-onnx's export, a GitHub release asset: copied next to the page by the Pages workflow (`models/`), SHA-256 checked | 0.5 MB | MIT |
| | `dpdfnet-baseline`, `dpdfnet2`, `dpdfnet4`, `dpdfnet8` | `Ceva-IP/DPDFNet` (16 kHz builds; the 8 and 48 kHz ones do not fit 16 kHz input) | 8.7–14.5 MB | Apache-2.0 |
| | `deepfilternet3` | `kimtos-labs/denoiser-dfn3` (torchDF's streaming graph, 48 kHz) | 16.1 MB | MIT OR Apache-2.0 |
| | `rnnoise` | `@shiguredo/rnnoise-wasm` 2025.1.5 from jsDelivr (its own WebAssembly build of Xiph's library) | 4.8 MB | BSD-3-Clause (wrapper Apache-2.0) |
| Identify the language first (Speech to text) | `voxlingua107-ecapa-int8` | `doanhlt/cabin-lid-ecapa-voxlingua107-onnx`; SpeechBrain's 60-band filterbank ported from `bolo.engines.lid_onnx` | 21.9 MB | Apache-2.0 |
| Voice activity (Conversation) | `silero-vad` (v4), `silero-vad-v5`, `silero-vad-v6`, `ten-vad`, `energy-vad` | `csukuangfj/vad`; Silero's repository (raw GitHub, v6); `TEN-framework/ten-vad` (its own WebAssembly build); nothing for the energy detector | 0–2.3 MB | MIT; TEN VAD: Apache-2.0 with Agora's conditions, not for commercial use |
| End of turn (Conversation) | `smart-turn-v3`, `smart-turn-v3.1`, `smart-turn-v3.0`, `smart-turn-v3-fp32` | `pipecat-ai/smart-turn-v3`; Whisper's log-mel ported from `bolo.audio.features` | 8.7–32.4 MB | BSD-2-Clause |
| Compare two voices (Speakers) | `eres2net-voxceleb` (default, calibrated), `campplus-voxceleb`, `campplus-zh-en`, `wespeaker-campplus-voxceleb`, `wespeaker-resnet34-voxceleb`, `ecapa-tdnn-voxceleb` | `csukuangfj/speaker-embedding-models`; `Pawls2theWall/equivl-ecapa-voxceleb-onnx` | 26–42 MB | Apache-2.0; WeSpeaker models CC-BY-4.0 |

Left out, with the reason shown next to the choices: the Demucs denoiser (PyTorch, non-commercial), ElevenLabs
Voice Isolator (cloud), MMS-LID (978 MB, non-commercial), AmberNet (NGC terms), the identifiers on the torch
plugin (Voxlect, SpeechBrain, mHuBERT: PyTorch), WebRTC VAD (no pinned WebAssembly build), MarbleNet VAD and
TitaNet (NeMo front end not ported), Namo (136–309 MB and a transcript decode at every pause), and everything
that needs a server: Voxtral, the speech-to-speech and audio-in models, the full-duplex ones (Moshi, Hibiki,
PersonaPlex, MiniCPM-o, FLM-Audio, Freeze-Omni).

The feature code follows the server: the same STFT, state and delay handling for each denoiser (checked by the
signal-to-noise ratio of a sample-aligned output, below), the server's `mix` for the strength, its live
half-second pieces with half a second of context, its routing for `model=auto` (`x-bolo-route` spelled the same;
a recogniser that is not downloaded is only suggested), its segmenter for frame-scoring detectors, Smart Turn's
8-second window, and its same / similar thresholds (0.50 / 0.30) for ERes2Net only. The filterbank ports are
held to the server's NumPy output by `ui/src/lib/browser-audio/dsp.test.ts`.

### WebAssembly or WebGPU: measured

Headless Chromium on the build machine (4 cores of an Intel Xeon @ 2.10 GHz, no GPU, load average 12 to 20
from other jobs). Headless Chromium has no hardware GPU here: WebGPU runs on SwiftShader (Vulkan emulated on the
CPU), so the WebGPU column says what per-op dispatch costs these graphs, not what a real GPU does; on a laptop
GPU the earlier recogniser measurements found small models 2 to 10 times faster on WebAssembly too. Full record:
`docs/measurements/browser-audio-models.json`. Real-time factor = processing time per second of audio.

| Model | WebAssembly | WebGPU (SwiftShader) | Other result |
| --- | --- | --- | --- |
| `gtcrn-simple` | 0.13 | 7.3 | SNR 5.0 → 13.5 dB (WebGPU 13.2) |
| `dpdfnet-baseline` | 0.20 | over 240 s for 5.5 s | SNR 5.0 → 13.3 dB |
| `dpdfnet2` | 0.40 | over 240 s | SNR 5.0 → 13.6 dB |
| `deepfilternet3` | 0.38 | 15.9 | SNR 5.0 → 11.7 dB |
| `rnnoise` | 0.15 | no WebGPU path | SNR 5.0 → 13.8 dB |
| `voxlingua107-ecapa-int8` | 0.14 | 4.0 | Hindi 0.99; English 0.46 on 5.5 s (0.81 on the 11 s clip; the server's code gives 0.47 and 0.81) |
| `silero-vad` / `-v5` / `-v6` | 0.039 / 0.015 / 0.018 | 3.0 / 2.4 / 2.5 | v5 and v6 mark different frames as speech on WebGPU |
| `ten-vad` | 0.017 | no WebGPU path | |
| `smart-turn-v3` / `-v3-fp32` | 0.11 / 0.12 | 4.5 / 3.9 | the 8-bit graph answers differently on WebGPU (0.23 against 0.38) |
| `eres2net-voxceleb` | 0.27 | over 240 s | same speaker 0.72, other speaker −0.06 |
| `campplus-voxceleb` | 0.085 | 5.1 | WebGPU's embeddings differ (other speaker 0.46 against −0.04) |
| `wespeaker-resnet34-voxceleb` | 0.19 | over 240 s | same 0.76, other −0.05 |
| `ecapa-tdnn-voxceleb` | 0.15 | over 240 s | same 0.66, other 0.01 |

Every model defaults to WebAssembly ("Auto"); WebGPU stays a choice in the page. Models run frame by frame
(denoisers, Silero) use one WebAssembly thread: with two, GTCRN took 0.70 instead of 0.13 here.
