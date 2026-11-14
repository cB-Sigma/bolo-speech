import { Store } from '@/lib/store'
import type { TranscriptState } from '@/lib/stt/transcriber'
import type { ChatBackend } from './engine'
import type { ChatMessage } from './models'

/**
 * A spoken conversation in the page: the recogniser's end of turn (the VAD
 * closing an utterance) sends the words to a language model, its reply is
 * cut into sentences as it streams, and each sentence is spoken as soon as it
 * is complete. Speaking while the assistant talks (barge-in) stops the reply,
 * the speech and the generation at once.
 */

export interface TurnLatency {
  /** End of speech to the final transcript (the recogniser's own figure). */
  sttMs: number | null
  /** Final transcript to the first token of the reply. */
  firstTokenMs: number | null
  /** Final transcript to the first sound of the reply. */
  firstAudioMs: number | null
}

export interface Turn {
  role: 'user' | 'assistant'
  text: string
  interrupted?: boolean
  latency?: TurnLatency
}

export interface ConversationState {
  status: 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'
  turns: Turn[]
  partial: string
  error: string | null
  bargeIns: number
}

export interface Speaker {
  /** Speak one sentence; resolves when its audio is queued. */
  say(sentence: string, signal: AbortSignal): Promise<void>
  /** Stop all speech now. */
  stop(): void
  /** Resolves when everything queued has been heard. */
  drained(): Promise<void>
}

export const SYSTEM_PROMPT =
  'You are a friendly voice assistant in a web page. Answer in one to three short spoken sentences, without lists, markdown or emoji.'

/** Complete sentences at the front of `text`, and the rest. */
export function takeSentences(text: string, final = false): { sentences: string[]; rest: string } {
  const out: string[] = []
  let rest = text
  for (;;) {
    const m = /^([\s\S]*?[.!?।…])(\s+|$)/.exec(rest)
    if (!m || (!m[2] && !final)) break
    const s = (m[1] as string).trim()
    if (s) out.push(s)
    rest = rest.slice(m[0].length)
  }
  if (final && rest.trim()) {
    out.push(rest.trim())
    rest = ''
  }
  return { sentences: out, rest }
}

export class Conversation {
  readonly store = new Store<ConversationState>({ status: 'idle', turns: [], partial: '', error: null, bargeIns: 0 })
  private abort: AbortController | null = null
  private seenFinals = 0
  private wasSpeaking = false

  private readonly llm: () => ChatBackend
  private readonly speaker: Speaker
  private readonly system: string

  constructor(llm: () => ChatBackend, speaker: Speaker, system = SYSTEM_PROMPT) {
    this.llm = llm
    this.speaker = speaker
    this.system = system
  }

  /** Feed every change of the recogniser's transcript state. */
  onTranscript(t: TranscriptState): void {
    const speaking = t.speech.length > 0 && t.speech.at(-1)?.t1 == null
    if (speaking && !this.wasSpeaking) this.onSpeechStart()
    this.wasSpeaking = speaking
    this.store.patch({ partial: t.partial?.text ?? '' })
    if (t.finals.length > this.seenFinals) {
      const fresh = t.finals.slice(this.seenFinals)
      this.seenFinals = t.finals.length
      const text = fresh
        .map((f) => f.text.trim())
        .filter(Boolean)
        .join(' ')
      if (text) void this.respond(text, fresh.at(-1)?.latencyMs ?? null)
    }
  }

  /** The visitor started talking: if the assistant is answering, stop it (barge-in). */
  onSpeechStart(): void {
    const s = this.store.get().status
    if (s !== 'thinking' && s !== 'speaking') return
    this.interrupt()
    this.store.patch({ bargeIns: this.store.get().bargeIns + 1 })
  }

  interrupt(): void {
    if (!this.abort) return
    this.abort.abort()
    this.abort = null
    this.speaker.stop()
    const turns = [...this.store.get().turns]
    const last = turns.at(-1)
    if (last?.role === 'assistant') turns[turns.length - 1] = { ...last, interrupted: true }
    this.store.patch({ turns, status: 'listening' })
  }

  async respond(userText: string, sttMs: number | null): Promise<void> {
    this.interrupt()
    const abort = new AbortController()
    this.abort = abort
    const t0 = performance.now()
    const history: ChatMessage[] = [
      { role: 'system', content: this.system },
      ...this.store
        .get()
        .turns.slice(-8)
        .map((t) => ({ role: t.role, content: t.text }) as ChatMessage),
      { role: 'user', content: userText },
    ]
    const latency: TurnLatency = { sttMs, firstTokenMs: null, firstAudioMs: null }
    this.store.patch({
      status: 'thinking',
      error: null,
      turns: [...this.store.get().turns, { role: 'user', text: userText }, { role: 'assistant', text: '', latency }],
    })
    const update = (patch: Partial<Turn>) => {
      if (abort.signal.aborted) return
      const turns = [...this.store.get().turns]
      const last = turns.at(-1) as Turn
      turns[turns.length - 1] = { ...last, ...patch, latency: { ...latency } }
      this.store.patch({ turns })
    }
    let pending = ''
    let text = ''
    let speech = Promise.resolve()
    const speak = (sentences: string[]) => {
      for (const s of sentences)
        speech = speech.then(async () => {
          if (abort.signal.aborted) return
          await this.speaker.say(s, abort.signal)
          if (latency.firstAudioMs == null && !abort.signal.aborted) {
            latency.firstAudioMs = Math.round(performance.now() - t0)
            this.store.patch({ status: 'speaking' })
            update({})
          }
        })
    }
    try {
      await this.llm().reply(
        history,
        (tok) => {
          if (abort.signal.aborted) return
          if (latency.firstTokenMs == null) latency.firstTokenMs = Math.round(performance.now() - t0)
          text += tok
          pending += tok
          const { sentences, rest } = takeSentences(pending)
          pending = rest
          speak(sentences)
          update({ text })
        },
        abort.signal,
      )
      if (abort.signal.aborted) return
      speak(takeSentences(pending, true).sentences)
      update({ text })
      await speech
      await this.speaker.drained()
      if (this.abort === abort) {
        this.abort = null
        this.store.patch({ status: 'listening' })
      }
    } catch (err) {
      if (abort.signal.aborted) return
      this.abort = null
      this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  }

  reset(): void {
    this.interrupt()
    this.seenFinals = 0
    this.store.set({ status: 'idle', turns: [], partial: '', error: null, bargeIns: 0 })
  }
}
