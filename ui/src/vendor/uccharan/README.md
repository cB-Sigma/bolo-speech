# uccharan (vendored copy)

The WebAssembly build of [uccharan](https://github.com/cb-Sigma/uccharan) 0.6.3, a permissively
licensed (Apache-2.0) phonemiser, text normaliser and transliterator. The console uses it to make
the phonemes that Piper and Kokoro voices read when they run in the browser, and in the text
playground; it contains no eSpeak NG code or data.

Copied unchanged from the npm package `uccharan@0.6.3` on 2026-11-16: `index.js`, `index.d.ts`,
`packs.js`, `dist/` (`uccharan_wasm_bg.wasm` sha256 `734b851d…e3bb43f4`), `data/*.ucd`,
`LICENSE`, `NOTICE`, `DATA_LICENSES.md`.

- The JavaScript and the 1.8 MiB WebAssembly module are part of the console bundle (loaded only
  by the speech-synthesis worker, the playground and, when a clip is generated, the "Clone a voice"
  page, whose models read text: it writes their numbers, dates and money out in words).
- `data/*.ucd` (word tables, 9.6 MB in all, see `NOTICE` and `DATA_LICENSES.md`) are not bundled:
  the voice build (`ui/scripts/voices/build.py`) publishes them next to the voices, with the
  CC BY-SA 4.0 transliteration packs from the uccharan release, and uccharan fetches each one the
  first time its language is used, checking it against the SHA-256 sums in `packs.js`.
- Since 0.6.1, Maithili, Bhojpuri and Magahi are read with the Hindi data (`hi.ucd`) and Belgian
  and Swiss French with the French word table, so no other files are needed; the playground offers
  all of uccharan's languages and turns romanised Maithili, Bhojpuri and Magahi into Devanagari with
  the Hindi pack. The SYSPIN voices for those languages read characters and do not use uccharan.
  The optional WikiPron packs are not published with the demo.
- 0.6.3 reads ordinals and dates in the European languages ("am 3. Mai" → "am dritten Mai", "le 1er
  mai" → "le premier mai"); 0.6.1 read the digits alone.

To update, copy the same files from a newer build and re-run the unit and e2e tests.
