# uccharan (vendored copy)

The WebAssembly build of [uccharan](https://github.com/cb-Sigma/uccharan) 0.1.0, a permissively
licensed (Apache-2.0) text-to-phoneme library. The console uses it to make the phonemes that Piper
voices read when they run in the browser; it contains no eSpeak NG code or data.

Copied unchanged from `bindings/wasm` of that project on 2026-11-13: `index.js`, `index.d.ts`,
`dist/uccharan_wasm.js`, `dist/uccharan_wasm.d.ts`, `dist/uccharan_wasm_bg.wasm`
(sha256 `1ef9fa51…0015`), `data/en-us.ucd` (sha256 `21987e99…b450`), `LICENSE`, `NOTICE`,
`DATA_LICENSES.md`.

- The JavaScript and the 254 KiB WebAssembly module are part of the console bundle (loaded only
  by the speech-synthesis worker).
- `data/en-us.ucd` (2.0 MiB, derived from CMUdict, see `NOTICE`) is not bundled: the voice build
  (`ui/scripts/voices/build.py`) publishes it next to the voices, and the worker fetches it with
  the Piper voice.

To update, copy the same files from a newer build and re-run the unit and e2e tests.
