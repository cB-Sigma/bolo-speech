/**
 * Text preparation for MOSS-TTS-Nano, ported from the model's Apache-2.0 reference code
 * (github.com/OpenMOSS/MOSS-TTS-Nano at 8b7bcc93):
 *
 * - `normalizeTtsText`: tts_robust_normalizer_single_script.py `normalize_tts_text`, the "robust" clean-up
 *   (URLs, e-mails, file names and versions kept whole; brackets to quotes; arrows, dashes and repeated
 *   punctuation tidied; spaces by script; a closing punctuation mark added). The reference also runs
 *   WeTextProcessing (numbers, dates and units spelt out in Chinese and English) when it is installed; that
 *   is a Python/FST package with no browser port, so digits reach the model as digits, as they do in the
 *   reference with WeText turned off.
 * - `splitVoiceCloneText` and `interChunkPauseSeconds`: onnx_tts_runtime.py, long text cut into chunks of at
 *   most 75 tokens at sentence ends, then clause marks, then by token budget, with a short pause between them.
 */

const CJK_CHARS = '\\u3400-\\u4dbf\\u4e00-\\u9fff\\u3040-\\u30ff'
const CJK = `[${CJK_CHARS}]`
const PROT = '___PROT\\d+___'
const DIGIT = '\\p{Nd}'

const re = (source: string, flags = 'gu') => new RegExp(source, flags)

const URL_RE = re('https?://[^\\s\\u3000，。！？；、）】》〉」』]+')
const EMAIL_RE = re('(?<![\\p{L}\\p{N}_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}(?![\\p{L}\\p{N}_.-])')
const MENTION_RE = re('(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}')
const REDDIT_RE = re('(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+')
const HASHTAG_RE = re('(?<![A-Za-z0-9_])#(?!\\s)[^\\s#]+')
const DOT_TOKEN_RE = re('(?<![A-Za-z0-9_])\\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+')
const FILELIKE_RE = re(
  '(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[./+:-])[A-Za-z0-9][A-Za-z0-9._/+:-]*(?![A-Za-z0-9_])',
)
const LATINISH = `(?:${PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)`
const ZERO_WIDTH_RE = re('[\\u200b-\\u200d\\ufeff]')
const TRAILING_CLOSERS = new Set([...'"\')]}）】》〉」』”’'])
const FLOW_ARROWS_RE = re('\\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\\s*')

/** The reference's robust TTS text clean-up (no reading of numbers, units or dates). */
export function normalizeTtsText(input: string): string {
  let text = baseCleanup(input)
  text = normalizeMarkdownAndLines(text)
  text = text.replace(FLOW_ARROWS_RE, '，')
  const protectedSpans: string[] = []
  for (const pattern of [URL_RE, EMAIL_RE, MENTION_RE, REDDIT_RE, HASHTAG_RE, DOT_TOKEN_RE, FILELIKE_RE])
    text = text.replace(pattern, (m) => `___PROT${protectedSpans.push(m) - 1}___`)
  text = text
    .split(re(`(${PROT})`))
    .map((part) => (re(`^${PROT}$`, 'u').test(part) ? part : part.replace(/_/g, ' ')))
    .join('')
  text = normalizeSpaces(text)
  text = normalizeStructuralPunctuation(text)
  text = normalizeRepeatedPunctuation(text)
  text = normalizeSpaces(text)
  protectedSpans.forEach((original, i) => {
    text = text.split(`___PROT${i}___`).join(original)
  })
  return ensureTerminalPunctuationByLine(text.trim())
}

function baseCleanup(text: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u3000/g, ' ')
    .replace(ZERO_WIDTH_RE, '')
    .replace(/[^\n\t ]/gu, (ch) => (/\p{C}/u.test(ch) ? '' : ch))
}

function normalizeMarkdownAndLines(input: string): string {
  const text = input.replace(re('\\[([^\\[\\]]+?)\\]\\((https?://[^)\\s]+)\\)'), '$1 $2')
  const lines: string[] = []
  for (const raw of text.split(re('\\n|\\u2028|\\u2029'))) {
    let line = raw.trim()
    if (!line) continue
    line = line
      .replace(/^#{1,6}\s+/u, '')
      .replace(/^>\s+/u, '')
      .replace(/^[-*+]\s+/u, '')
      .replace(re(`^${DIGIT}+[.)]\\s+`, 'u'), '')
    lines.push(line)
  }
  if (!lines.length) return ''
  const merged = [lines[0] as string]
  for (const line of lines.slice(1)) {
    merged[merged.length - 1] = ensureTerminalPunctuation(merged[merged.length - 1] as string)
    merged.push(line)
  }
  return merged.join('')
}

function normalizeSpaces(input: string): string {
  let text = input.replace(/[ \t\r\f\v]+/g, ' ')
  text = text.replace(re(`(${CJK})\\s+(?=${CJK})`), '$1')
  text = text.replace(re(`(${CJK})\\s+(?=${DIGIT})`), '$1')
  text = text.replace(re(`(${DIGIT})\\s+(?=${CJK})`), '$1')
  text = text.replace(re(`(${CJK})(?=(${LATINISH}))`), '$1 ')
  text = text.replace(re(`((${LATINISH}))(?=${CJK})`), '$1 ')
  text = text.replace(/ {2,}/g, ' ')
  text = text.replace(re('\\s+([，。！？；：、”’」』】）》])'), '$1')
  text = text.replace(re('([（【「『《“‘])\\s+'), '$1')
  text = text.replace(re('([，。！？；：、])\\s*'), '$1')
  text = text.replace(re('\\s+([,.;!?])'), '$1')
  return text.replace(/ {2,}/g, ' ').trim()
}

function normalizeStructuralPunctuation(input: string): string {
  let text = input.replace(re('\\[\\s*([^\\[\\]]+?)\\s*\\]'), '"$1"')
  text = text.replace(re('\\{\\s*([^{}]+?)\\s*\\}'), '"$1"')
  text = text.replace(re('[【〖『「]\\s*([^】〗』」]+?)\\s*[】〗』」]'), '"$1"')
  // A title in 《》 on its own (a headline) loses the brackets; one inside a sentence keeps them.
  text = text.replace(re(`(^|[。！？!?；;]\\s*)《([^》]+)》(?=\\s*(?:${PROT}|[—–―-]{2,}|$|[。！？!?；;，,]))`), '$1$2')
  text = text.replace(FLOW_ARROWS_RE, '，')
  return text.replace(re('\\s*(?:—|–|―|-){2,}\\s*'), '。')
}

function normalizeRepeatedPunctuation(input: string): string {
  return input
    .replace(re('(?:\\.{3,}|…{2,}|……+)'), '。')
    .replace(re('[。．]{2,}'), '。')
    .replace(re('[，,]{2,}'), '，')
    .replace(re('[!！]{2,}'), '！')
    .replace(re('[?？]{2,}'), '？')
    .replace(re('[!?！？]{2,}'), (s) => {
      const q = /[?？]/.test(s)
      const e = /[!！]/.test(s)
      return q && e ? '？！' : q ? '？' : '！'
    })
}

function ensureTerminalPunctuation(text: string): string {
  if (!text) return text
  const chars = Array.from(text)
  let i = chars.length - 1
  while (i >= 0 && /\s/u.test(chars[i] as string)) i--
  while (i >= 0 && TRAILING_CLOSERS.has(chars[i] as string)) i--
  if (i >= 0 && /\p{P}/u.test(chars[i] as string)) return text
  return `${text}。`
}

function ensureTerminalPunctuationByLine(text: string): string {
  if (!text) return text
  return text
    .split('\n')
    .map((line) => (line.trim() ? ensureTerminalPunctuation(line.trim()) : ''))
    .join('\n')
    .trim()
}

// ---- Chunking of long text (onnx_tts_runtime.py) ----

const SENTENCE_END = new Set([...'.!?。！？；;'])
const CLAUSE_SPLIT = new Set([...',，、；;：:'])
const CLOSING = new Set([...'"\'”’)]}）】》」』'])

export const MAX_CHUNK_TOKENS = 75

export function containsCjk(text: string): boolean {
  return /[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]/u.test(text)
}

const words = (text: string) => text.split(/\s+/u).filter(Boolean)

/** Text made ready for cutting into sentences: one line, capitalised, closed with a full stop. */
export function prepareForChunking(input: string): string {
  let text = input.trim()
  if (!text) throw new Error('Text cannot be empty')
  text = text.replace(/[\r\n]/g, ' ')
  while (text.includes('  ')) text = text.replace(/ {2}/g, ' ')
  const chars = Array.from(text)
  const last = chars[chars.length - 1] as string
  if (containsCjk(text)) return SENTENCE_END.has(last) ? text : `${text}。`
  const first = chars[0] as string
  if (/\p{Ll}/u.test(first)) text = first.toUpperCase() + chars.slice(1).join('')
  if (/[\p{L}\p{N}]/u.test(last)) text += '.'
  if (words(text).length < 5) text = `        ${text}`
  return text
}

/** Cut after each mark of `marks` (and the closing quotes or brackets right after it). */
export function splitByPunctuation(text: string, marks: Set<string>): string[] {
  const chars = Array.from(text)
  const out: string[] = []
  let cur: string[] = []
  let i = 0
  while (i < chars.length) {
    const ch = chars[i] as string
    cur.push(ch)
    if (marks.has(ch)) {
      let j = i + 1
      while (j < chars.length && CLOSING.has(chars[j] as string)) cur.push(chars[j++] as string)
      const sentence = cur.join('').trim()
      if (sentence) out.push(sentence)
      cur = []
      while (j < chars.length && /\s/u.test(chars[j] as string)) j++
      i = j
      continue
    }
    i++
  }
  const tail = cur.join('').trim()
  if (tail) out.push(tail)
  return out
}

function joinParts(left: string, right: string): string {
  if (!left) return right
  if (!right) return left
  return containsCjk(left) || containsCjk(right) ? left + right : `${left} ${right}`
}

/** Pieces of at most `max` tokens, cut at a space or punctuation mark near the limit when there is one. */
export function splitByTokenBudget(input: string, max: number, count: (t: string) => number): string[] {
  let rest = Array.from(input.trim())
  const out: string[] = []
  const preferred = new Set([...CLAUSE_SPLIT, ...SENTENCE_END, ' '])
  while (rest.length) {
    const restText = rest.join('')
    if (count(restText) <= max) {
      out.push(restText)
      break
    }
    let low = 1
    let high = rest.length
    let best = 1
    while (low <= high) {
      const mid = Math.floor((low + high) / 2)
      const candidate = rest.slice(0, mid).join('').trim()
      if (!candidate) {
        low = mid + 1
        continue
      }
      if (count(candidate) <= max) {
        best = mid
        low = mid + 1
      } else high = mid - 1
    }
    let cut = best
    let pref = -1
    for (let k = best - 1; k > Math.max(-1, best - 25); k--)
      if (preferred.has(rest[k] as string)) {
        pref = k + 1
        break
      }
    if (pref > 0) cut = pref
    let piece = rest.slice(0, cut).join('').trim()
    if (!piece) {
      piece = rest.slice(0, best).join('').trim()
      cut = best
    }
    out.push(piece)
    rest = Array.from(rest.slice(cut).join('').trim())
  }
  return out
}

/**
 * The text as the model is given it, chunk by chunk: one chunk (the text unchanged) when it fits in `max`
 * tokens, else whole sentences joined up to `max` tokens, a long sentence cut at clause marks, then by count.
 */
export function splitVoiceCloneText(input: string, count: (t: string) => number, max = MAX_CHUNK_TOKENS): string[] {
  const text = input.trim()
  if (!text) return []
  const limit = Math.max(1, Math.floor(max))
  const prepared = prepareForChunking(text)
  const sentences = splitByPunctuation(prepared, SENTENCE_END)
  const slices: [number, string][] = []
  for (const s of sentences.length ? sentences : [prepared.trim()]) {
    const sentence = s.trim()
    if (!sentence) continue
    const n = count(sentence)
    if (n <= limit) {
      slices.push([n, sentence])
      continue
    }
    let clauses = splitByPunctuation(sentence, CLAUSE_SPLIT)
    if (clauses.length <= 1) clauses = [sentence]
    for (const c of clauses) {
      const clause = c.trim()
      if (!clause) continue
      const m = count(clause)
      if (m <= limit) {
        slices.push([m, clause])
        continue
      }
      for (const p of splitByTokenBudget(clause, limit, count)) {
        const piece = p.trim()
        if (piece) slices.push([count(piece), piece])
      }
    }
  }
  const chunks: string[] = []
  let cur = ''
  let n = 0
  for (const [k, s] of slices) {
    if (!cur) {
      cur = s
      n = k
    } else if (n + k > limit) {
      chunks.push(cur.trim())
      cur = s
      n = k
    } else {
      cur = joinParts(cur, s)
      n = count(cur)
    }
  }
  if (cur) chunks.push(cur.trim())
  return chunks.length > 1 ? chunks : [text]
}

/** Silence between two chunks: a little longer after a short one. */
export function interChunkPauseSeconds(chunk: string): number {
  return words(chunk.trim()).length <= 4 ? 0.4 : 0.24
}
