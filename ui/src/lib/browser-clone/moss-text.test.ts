import { describe, expect, it } from 'vitest'
import {
  containsCjk,
  interChunkPauseSeconds,
  normalizeTtsText,
  prepareForChunking,
  splitByPunctuation,
  splitByTokenBudget,
  splitVoiceCloneText,
} from './moss-text'

/** Cases from the reference normaliser's own test list (tts_robust_normalizer_single_script.py). */
const NORMALIZER_CASES: [string, string][] = [
  ['别把 .env、.npmrc、.gitignore 提交上去。', '别把 .env、.npmrc、.gitignore 提交上去。'],
  ['index.d.ts 里也有同样的问题。', 'index.d.ts 里也有同样的问题。'],
  ['3.0.0-rc.1 还不能上生产。', '3.0.0-rc.1 还不能上生产。'],
  ['联系邮箱：ops+tts@example.ai', '联系邮箱：ops+tts@example.ai。'],
  ['#张雪峰#张雪峰[话题]#张雪峰事件', '#张雪峰#张雪峰[话题]#张雪峰事件。'],
  ['关注@biscuit0228_并转发#thetime_tbs', '关注 @biscuit0228_ 并转发 #thetime_tbs。'],
  ['[S1]你好。[S2]收到。', '"S1"你好。"S2"收到。'],
  ['请模仿 {whisper} 的语气说“别出声”。', '请模仿 "whisper" 的语气说“别出声”。'],
  ['〖重磅〗《新品发布》——现在开始！', '"重磅"《新品发布》。现在开始！'],
  ['【公告】今天 20:00 维护——预计 30 分钟。', '"公告"今天20:00维护。预计30分钟。'],
  ['请求接入 -> 身份与策略判定 -> 域服务处理', '请求接入，身份与策略判定，域服务处理。'],
  ['我喜欢《哈姆雷特》这本书。', '我喜欢《哈姆雷特》这本书。'],
  ['真的假的？？？！！！', '真的假的？！'],
  ['这个包把 app.js.map 也发上去了......太离谱了！！！', '这个包把 app.js.map 也发上去了。太离谱了！'],
  ['This   is   a   test.', 'This is a test.'],
  [
    '\u8fd9 \u662f\u3000\u4e00 \u6bb5  \u542b\u6709\u591a\u79cd\u7a7a\u767d\u7684\u6587\u672c\u3002',
    '\u8fd9\u662f\u4e00\u6bb5\u542b\u6709\u591a\u79cd\u7a7a\u767d\u7684\u6587\u672c\u3002',
  ],
  ['今天update到v2.3.1了', '今天 update 到 v2.3.1 了。'],
  ['foo_bar', 'foo bar。'],
  [
    '详情见 [release note](https://github.com/example/release)',
    '详情见 release note https://github.com/example/release。',
  ],
  [
    '# I made a free open source app to help with markdown files',
    'I made a free open source app to help with markdown files。',
  ],
  ['- 修复 .map 泄露\n- 发布 v2.3.1', '修复 .map 泄露。发布 v2.3.1。'],
  ['第一行\n第二行\n第三行', '第一行。第二行。第三行。'],
  ['他说"你好"', '他说"你好"。'],
  ['第一行\n第二行。', '第一行。第二行。'],
  ['\u8be6\u89c1 https://x.com/\u200bSafety', '\u8be6\u89c1 https://x.com/Safety\u3002'],
]

/** A stand-in tokenizer: one token per word and per punctuation mark, one per CJK character. */
const count = (t: string) =>
  (t.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu) ?? []).reduce((a, w) => a + (containsCjk(w) ? [...w].length : 1), 0)

describe('MOSS-TTS-Nano text preparation', () => {
  it('cleans text up as the reference normaliser does', () => {
    for (const [input, expected] of NORMALIZER_CASES) {
      expect(normalizeTtsText(input), input).toBe(expected)
      // and doing it twice changes nothing
      expect(normalizeTtsText(expected), expected).toBe(expected)
    }
  })

  it('leaves numbers as digits (the WeText step is not in the browser)', () => {
    expect(normalizeTtsText('Call me at 10:30 on 2024-05-01')).toBe('Call me at 10:30 on 2024-05-01。')
  })

  it('prepares text for chunking: one line, capital, full stop, padding for short ones', () => {
    expect(prepareForChunking('hello there\nmy friend how are you')).toBe('Hello there my friend how are you.')
    expect(prepareForChunking('hi')).toBe('        Hi.')
    expect(prepareForChunking('你好')).toBe('你好。')
    expect(() => prepareForChunking('  ')).toThrow(/empty/)
  })

  it('cuts after sentence marks, keeping closing quotes with them', () => {
    expect(splitByPunctuation('He said "Stop!" Then left. Ok', new Set(['.', '!']))).toEqual([
      'He said "Stop!"',
      'Then left.',
      'Ok',
    ])
  })

  it('cuts a long run without punctuation by token budget, at a space near the limit', () => {
    const text = Array.from({ length: 30 }, (_, i) => `w${i}`).join(' ')
    const pieces = splitByTokenBudget(text, 10, count)
    expect(pieces.every((p) => count(p) <= 10)).toBe(true)
    expect(pieces.join(' ')).toBe(text)
  })

  it('keeps text that fits in one chunk as it is', () => {
    expect(splitVoiceCloneText('short text', count)).toEqual(['short text'])
  })

  it('packs whole sentences into chunks of at most the budget, clauses when a sentence is too long', () => {
    const sentence = (i: number) => `This is sentence ${i} and it has a few more words.`
    const text = Array.from({ length: 12 }, (_, i) => sentence(i)).join(' ')
    const chunks = splitVoiceCloneText(text, count, 30)
    expect(chunks.length).toBeGreaterThan(3)
    expect(chunks.every((c) => count(c) <= 30)).toBe(true)
    expect(chunks.join(' ')).toBe(text)
    const long = `${Array.from({ length: 8 }, (_, i) => `clause number ${i} goes here`).join(', ')}.`
    const parts = splitVoiceCloneText(`${long} ${long}`, count, 20)
    expect(parts.every((c) => count(c) <= 20)).toBe(true)
  })

  it('joins Chinese sentences without spaces', () => {
    const zh = '今天天气很好。'.repeat(10)
    const chunks = splitVoiceCloneText(zh, count, 20)
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.join('')).toBe(zh)
  })

  it('pauses a little longer after a short chunk', () => {
    expect(interChunkPauseSeconds('Yes, indeed.')).toBe(0.4)
    expect(interChunkPauseSeconds('This one has more than four words.')).toBe(0.24)
  })
})
