# Images in this directory

Everything here is either a real screenshot of the running web console or a brand asset rendered
from the sources described in `brand/brand.md`. No screenshot is mocked up or edited.

Screenshots are taken with `ui/scripts/screenshots.mjs` (Playwright, headless Chromium). The
numbers in them are measurements made on the build machine on 2026-11-13: 4 cores of Intel(R) Xeon(R) Processor @ 2.10GHz,
Linux x86_64, no GPU. They will differ on other hardware.

"Real server" below means `bolo serve` 0.1.0 with the sherpa-onnx engine on that machine.
"In-browser demo" means the static demo build (`npm run build:demo`) served from a plain static
file server with no special headers, the way GitHub Pages serves it, with speech recognition
running in the page (Moonshine Tiny, ONNX Runtime Web, WebAssembly). The speech recording used is an
11-second excerpt of John F. Kennedy's 1961 inaugural address (public domain); `en-long` is one of
the server's own built-in samples.

## Console screenshots

| File | Size | Theme | Taken against | What it shows |
| --- | --- | --- | --- | --- |
| `console-playground.png` | 1440 px, 2× | dark | real server, `moonshine-tiny-en` | Speech-to-text playground after streaming the recording over the WebSocket at real-time pace: transcript, VAD segments, latency timeline, budget and split. The latency target was set to 800 ms for this picture (the default is 300 ms). |
| `console-light.png` | 1440 px, 2× | light | real server, `moonshine-tiny-en` | The same screen in the light theme (a separate run, so figures differ slightly). |
| `console-playground.gif` | 960 px, 10 fps, about 16 s | dark | real server, `moonshine-tiny-en` | Animated capture of a live transcription of the server's `en-long` sample: partial text firming up into final text, the level strip, and the latency timeline filling in. Frames are consecutive Playwright screenshots of the real page (sidebar cropped), assembled with ffmpeg; nothing is sped up. |
| `console-timeline.png` | panel crop, 2× | dark | real server, `moonshine-tiny-en` | Close-up of the latency timeline panel after a run. |
| `console-compare.png` | 1440 px, 2× | dark | real server **and** in-browser recogniser | Compare mode: the same recording through the server and through the in-browser model, with both timelines. |
| `console-phone.png` | 390 px, 3× | dark | real server, `moonshine-tiny-en` | The playground at phone width. |
| `console-latency.png` | 1440 px, 2× | dark | real server | Latency dashboard while a script sent real requests for about a minute: HTTP transcriptions (`moonshine-tiny-en`, `whisper-tiny`), real-time WebSocket streams (`moonshine-tiny-en`, `zipformer-en-20m-streaming`) and HTTP synthesis (`kokoro-v1.0-int8`). |
| `console-models.png` | 1440 px, 2× | dark | real server | The catalog as the server reported it. |
| `console-tts.png` | 1440 px, 2× | dark | real server, `kokoro-v1.0-int8` | Text-to-speech playground after "Speak, then transcribe": streamed synthesis, time to first audio, the chunk timeline and the round trip through `moonshine-tiny-en`. Kokoro runs slower than real time on this CPU, so the playback underrun shown is genuine. |
| `console-benchmark.png` | 1440 px, 2× | dark | real server, `moonshine-tiny-en` | A concurrency sweep (1, 2, 4, 8; 5 s per level) run through `POST /api/v1/bench`. |
| `console-demo.png` | 1440 px, 2× | dark | in-browser demo | The live demo as a first-time visitor gets it (default 800 ms target), after pressing "Play a sample": recognition ran in the page on 4 WebAssembly threads (cross-origin isolation supplied by the demo's service worker). No server was involved. |
| `console-components.png` | 1440 px, 2× | dark | real server, `piper-en-us-amy-low` | Text-to-speech playground with a voice that needs eSpeak NG while it is not installed: the inline notice and install button. Taken against the real server code running from a Python environment without the eSpeak NG package (the build machine's usual environment already has it). |
| `console-components-dialog.png` | 1440 px, 2× | dark | real server | The confirmation shown before installing: licence, link, and what will happen. Confirming it really installed the package into that scratch environment in about a second (from a warm package cache), and the voice then spoke without a reload. |

In the compare and demo pictures the model files and the ONNX Runtime binary were fetched from a
local mirror on a second origin that sends the same CORS header as the public CDNs, because the
build machine's browser has no direct internet access. The code path is otherwise the one a visitor
gets.

## Brand

| File | What it is |
| --- | --- |
| `brand/logo-mark.svg`, `brand/logo-mark-mono.svg` | The mark, in its tile and as bare strokes. |
| `brand/logo-wordmark-dark.svg`, `brand/logo-wordmark-light.svg` | Mark and name for dark and light backgrounds; text is outlined. |
| `brand/banner-dark.png`, `brand/banner-light.png` | README hero banners, 3200 × 960 px (1600 × 480 at 2×). |
| `brand/social-preview.png` | 1280 × 640 social preview / Open Graph image. |
| `brand/button-live-demo.png` | "Try the live demo" button for the README (466 × 104 px, shown at half size). |
| `brand/favicon/` | `favicon.svg`, `favicon.ico`, 16/32/48 px PNGs, `apple-touch-icon.png`, 192 and 512 px icons. |
| `brand/brand.md` | Colours, spacing, do and don't. |

Brand images are rendered from SVG/HTML with headless Chromium; they are drawings, not screenshots.

## Retaking the screenshots

```bash
cd ui
node scripts/screenshots.mjs --base http://127.0.0.1:8000 --out ../docs/assets \
  --speech path/to/speech.wav --stt moonshine-tiny-en --tts kokoro-v1.0-int8 --playground-target 800 \
  --demo http://127.0.0.1:4174/bolo        # after: npm run build:demo && npm run serve:demo
```

