# uccharan (vendored copy)

The WebAssembly build of [uccharan](https://github.com/cb-Sigma/uccharan) 0.4.0, a permissively
licensed (Apache-2.0) phonemiser, text normaliser and transliterator. The console uses it to make
the phonemes that Piper and Kokoro voices read when they run in the browser, and in the text
playground; it contains no eSpeak NG code or data.

Copied unchanged from the npm package `uccharan@0.4.0` on 2026-11-14: `index.js`, `index.d.ts`,
`packs.js`, `dist/` (`uccharan_wasm_bg.wasm` sha256 `b999c594…657d5ddc`), `data/*.ucd`,
`LICENSE`, `NOTICE`, `DATA_LICENSES.md`.

- The JavaScript and the 1.6 MiB WebAssembly module are part of the console bundle (loaded only
  by the speech-synthesis worker and the playground).
- `data/*.ucd` (word tables, 9 MB in all, see `NOTICE` and `DATA_LICENSES.md`) are not bundled:
  the voice build (`ui/scripts/voices/build.py`) publishes them next to the voices, with the
  CC BY-SA 4.0 transliteration packs from the uccharan release, and uccharan fetches each one the
  first time its language is used, checking it against the SHA-256 sums in `packs.js`.
- In this version Maithili, Bhojpuri and Magahi look for data files of their own that the package
  does not ship; the console does not ask uccharan for those languages (their voices read characters).

To update, copy the same files from a newer build and re-run the unit and e2e tests.
