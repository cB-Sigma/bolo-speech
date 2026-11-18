/**
 * tiny-voca (Muhammed Nazeem, Apache-2.0) in the browser: the authors' ONNX export, voices F2 and M2, 24 kHz.
 *
 * Its phonemes come from dizione, the authors' phonemiser (Apache-2.0), which exists only in Python: it runs here
 * unchanged under Pyodide (CPython on WebAssembly, MPL-2.0), as the authors' own browser studio does. dizione's
 * source is fetched from its repository at the pinned commit, its data (the dictionary and the small model for unknown
 * words) from the voice's repository at the pinned revision, exactly the files the server uses.
 */

/** The model repository, pinned. */
export const TINY_VOCA_REVISION = '1e73262226dbf4ef02e9df277e81aaa1db5ce16c'
export const TINY_VOCA_BASE = `https://huggingface.co/n4ze3m/tiny-voca/resolve/${TINY_VOCA_REVISION}/`
/** dizione at the commit the model's front end was built with (as bolo's server copy, bolo/vendor/dizione). */
export const DIZIONE_COMMIT = '8cd889e4de7be6bd1dd2e6891344a0729fa14dd4'
export const DIZIONE_BASE = `https://cdn.jsdelivr.net/gh/n4ze3m/dizione@${DIZIONE_COMMIT}/src/dizione/`
/** Pyodide, pinned (its numpy is the one package dizione needs). */
export const PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'

/** dizione's package files. */
export const DIZIONE_FILES = [
  '__init__.py', 'align.py', 'convert.py', 'engine.py', 'g2p.py', 'homographs.py', 'morphology.py',
  'normalize.py', 'sentence.py', 'symbols.py', 'tables.py',
  'data/abbreviations.txt', 'data/chat.txt', 'data/endings.txt', 'data/exceptions.txt', 'data/function_words.txt',
  'data/homograph_fixes.txt', 'data/homograph_rules.txt', 'data/letters.txt', 'data/phrases.txt', 'data/units.txt',
] // prettier-ignore

/** The voice's phonemiser data (the model repository's phonemizer/ folder). */
export const DIZIONE_DATA_FILES = [
  'lexicon.tsv', 'lexicon.json', 'align.json', 'g2p.json', 'g2p.npz', 'homographs.json', 'sources.json',
] // prettier-ignore

/** The model reads one utterance of at most 512 tokens; the reference keeps a margin of 12. */
export const TINY_VOCA_TOKEN_LIMIT = 500
/** The loudest sample of the audio (the model can peak a little above 1.0). */
export const TINY_VOCA_PEAK = 0.99

const SPACE = /[\s\uFEFF]+/
const TEXT_END = /[\s\uFEFF]+$/
const CLAUSE_END = /(?<=[,;:—])\s+/
const CLOSING = '"\'”’)]»'

/**
 * `text` with a full stop when it ends without a mark: without one the voice breaks off the last word (the reference's
 * `tiny_voca.infer.finished`).
 */
export function finished(text: string): string {
  const body = text.replace(TEXT_END, '')
  if (!body) return text
  const words = body.split(SPACE)
  const last = words[words.length - 1] ?? ''
  let marked: boolean
  if ([...last].some((c) => /[\p{L}\p{N}]/u.test(c))) {
    let stripped = last
    while (stripped && CLOSING.includes(stripped.slice(-1))) stripped = stripped.slice(0, -1)
    marked = '.!?…,;:'.includes(stripped.slice(-1)) && stripped.slice(-1) !== ''
  } else {
    marked = [...last].some((c) => '.!?…'.includes(c)) || last === ',' || last === ';' || last === ':'
  }
  return marked ? text : `${body}.`
}

/** `text` in smaller pieces: after commas and the like, else in two between words; [] for one word. */
export function shorter(text: string): string[] {
  const pieces = text.split(CLAUSE_END)
  if (pieces.length > 1) return pieces
  const words = text.split(' ')
  if (words.length < 2) return []
  const half = Math.ceil(words.length / 2)
  return [words.slice(0, half).join(' '), words.slice(half).join(' ')]
}

/** The audio scaled down as a whole when its loudest sample is above the peak. */
export function limitPeak(audio: Float32Array, peak = TINY_VOCA_PEAK): Float32Array {
  let loudest = 0
  for (const x of audio) loudest = Math.max(loudest, Math.abs(x))
  if (loudest <= peak) return audio
  const k = peak / loudest
  return audio.map((x) => x * k)
}
