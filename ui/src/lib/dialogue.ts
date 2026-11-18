/** Multi-speaker dialogues: `POST /v1/audio/dialogue` and the helpers of the Dialogue builder. */

export interface DialogueTurn {
  speaker: string
  text: string
}

export type DialogueMode = 'auto' | 'native' | 'stitched'

export interface DialogueBody {
  model?: string
  turns: DialogueTurn[]
  voices: Record<string, string>
  mode?: DialogueMode
  pause_ms?: number
  instructions?: string
  response_format?: string
}

export interface DialogueSegment {
  turn: number
  speaker: string
  start: number
  end: number
}

/** The reference clips a one-take dialogue can use for a speaker (bolo's samples). */
export const REFERENCE_CLIPS = ['en-short', 'en-long', 'hi-short']

/** The model speaks a whole script in one take (`features` has `dialogue`). */
export function speaksDialogues(m: { features: string[] } | null | undefined): boolean {
  return Boolean(m?.features.includes('dialogue'))
}

/** The speakers in order of first appearance (names trimmed; empty names left out). */
export function speakersOf(turns: DialogueTurn[]): string[] {
  const out: string[] = []
  for (const t of turns) {
    const name = t.speaker.trim()
    if (name && !out.includes(name)) out.push(name)
  }
  return out
}

/** `Name: text` lines to turns; a line without a name continues the previous turn. */
export function parseScript(script: string): DialogueTurn[] {
  const turns: DialogueTurn[] = []
  for (const raw of script.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line) continue
    const m = /^([^:]{1,40}?)\s*:\s*(.+)$/.exec(line)
    const name = m?.[1]
    const said = m?.[2]
    const last = turns[turns.length - 1]
    if (name && said && !/^(https?|\[)/.test(name)) turns.push({ speaker: name.trim(), text: said.trim() })
    else if (last) last.text += ` ${line}`
  }
  return turns
}

/** Turns back to a `Name: text` script. */
export function toScript(turns: DialogueTurn[]): string {
  return turns.map((t) => `${t.speaker}: ${t.text}`).join('\n')
}

/** The request body: turns with a speaker and text, and voices only for the speakers that are left. */
export function dialogueBody(
  model: string | undefined,
  turns: DialogueTurn[],
  voices: Record<string, string>,
  opts: { mode?: DialogueMode; pauseMs?: number; instructions?: string } = {},
): DialogueBody {
  const kept = turns.map((t) => ({ speaker: t.speaker.trim(), text: t.text.trim() })).filter((t) => t.speaker && t.text)
  const speakers = speakersOf(kept)
  const chosen: Record<string, string> = {}
  for (const s of speakers) if (voices[s]) chosen[s] = voices[s]
  return {
    model,
    turns: kept,
    voices: chosen,
    mode: opts.mode && opts.mode !== 'auto' ? opts.mode : undefined,
    pause_ms: opts.pauseMs,
    instructions: opts.instructions?.trim() || undefined,
    response_format: 'wav',
  }
}

/** `x-bolo-segments` (stitched dialogues): where each turn starts and ends, in seconds. */
export function parseSegments(header: string | null): DialogueSegment[] {
  if (!header) return []
  try {
    const parsed: unknown = JSON.parse(header)
    return Array.isArray(parsed) ? (parsed as DialogueSegment[]) : []
  } catch {
    return []
  }
}
