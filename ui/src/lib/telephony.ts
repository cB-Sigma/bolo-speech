import type { CallTurn, PhoneAgent, PhoneAgentUpdate, TelephonyProvider } from './api/types'

export const DOCS_TELEPHONY = 'https://github.com/cB-Sigma/bolo-speech/blob/main/docs/telephony.md'
export const AGENT_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/

export const OUTCOME_LABEL: Record<string, string> = {
  caller_hangup: 'Caller hung up',
  agent_hangup: 'Agent hung up',
  transferred: 'Transferred',
  max_duration: 'Reached the longest call length',
  no_response: 'Caller went silent',
  voicemail: 'Voicemail',
  no_answer: 'No answer',
  busy: 'Busy',
  failed: 'Failed',
  canceled: 'Cancelled',
  error: 'Error',
}

/** Outcomes that mean something went wrong (shown in the critical colour). */
export const BAD_OUTCOMES = new Set(['failed', 'error'])

export function outcomeLabel(outcome: string | null | undefined, status?: string): string {
  if (!outcome) return status === 'in_progress' ? 'In progress' : status === 'ringing' ? 'Ringing' : 'Unknown'
  return OUTCOME_LABEL[outcome] ?? outcome
}

/** The state of a provider in one word, and whether calls can reach it. */
export function providerState(p: TelephonyProvider): { label: string; ready: boolean } {
  if (p.exposed && p.configured) return { label: 'Ready', ready: true }
  if (p.exposed) return { label: 'Open without validation', ready: true }
  return { label: p.configured ? 'Closed: settings missing' : 'Not set up', ready: false }
}

/** The pieces of a turn's latency in the order they happen, with what they mean. */
export const LATENCY_PARTS: { key: keyof CallTurn['latency']; label: string; hint: string }[] = [
  { key: 'endpoint_ms', label: 'End of speech', hint: 'silence the turn detector waits for' },
  { key: 'stt_ms', label: 'Speech to text', hint: 'transcribing the turn' },
  { key: 'llm_first_token_ms', label: 'LLM first token', hint: 'the language model starts answering' },
  { key: 'tts_first_audio_ms', label: 'TTS first audio', hint: 'the first sentence is synthesised' },
  { key: 'send_ms', label: 'Sent', hint: 'the first audio frame goes to the provider' },
]

/** The parts of a turn that were measured, with their share of the whole (for a stacked bar). */
export function latencyShares(turn: CallTurn): { key: string; label: string; ms: number; share: number }[] {
  const parts = LATENCY_PARTS.map((p) => ({ key: p.key, label: p.label, ms: turn.latency[p.key] })).filter(
    (p): p is { key: keyof CallTurn['latency']; label: string; ms: number } => typeof p.ms === 'number',
  )
  const total = parts.reduce((sum, p) => sum + p.ms, 0)
  return parts.map((p) => ({ ...p, share: total > 0 ? p.ms / total : 0 }))
}

/** What the agent dialog holds: every field as the text or number the form shows. */
export interface AgentForm {
  name: string
  instructions: string
  greeting: string
  voice: string
  language: string
  sttModel: string
  ttsModel: string
  llmModel: string
  pipeline: string
  maxDurationS: string
  maxDurationMessage: string
  silenceTimeoutS: string
  silencePrompt: string
  silencePrompts: string
  goodbye: string
  transferNumber: string
  transferMessage: string
  recordingConsent: string
  voicemail: string
  voicemailMessage: string
  tools: string
}

export function agentForm(agent: PhoneAgent | null): AgentForm {
  return {
    name: agent?.name ?? '',
    instructions: agent?.instructions ?? '',
    greeting: agent?.greeting ?? '',
    voice: agent?.voice ?? '',
    language: agent?.language ?? '',
    sttModel: agent?.stt_model ?? '',
    ttsModel: agent?.tts_model ?? '',
    llmModel: agent?.llm_model ?? '',
    pipeline: agent?.pipeline ?? '',
    maxDurationS: String(agent?.max_duration_s ?? 600),
    maxDurationMessage: agent?.max_duration_message ?? '',
    silenceTimeoutS: String(agent?.silence_timeout_s ?? 10),
    silencePrompt: agent?.silence_prompt ?? 'Are you still there?',
    silencePrompts: String(agent?.silence_prompts ?? 2),
    goodbye: agent?.goodbye ?? '',
    transferNumber: agent?.transfer_number ?? '',
    transferMessage: agent?.transfer_message ?? '',
    recordingConsent: agent?.recording_consent ?? '',
    voicemail: agent?.voicemail ?? 'hangup',
    voicemailMessage: agent?.voicemail_message ?? '',
    tools: agent?.tools.length ? JSON.stringify(agent.tools, null, 2) : '',
  }
}

const text = (value: string): string | null => value.trim() || null

/** The request body for a form, or the first thing wrong with it. */
export function agentUpdate(form: AgentForm): { body: PhoneAgentUpdate; problem: string | null } {
  const body: PhoneAgentUpdate = {
    instructions: text(form.instructions),
    greeting: text(form.greeting),
    voice: text(form.voice),
    language: text(form.language),
    stt_model: text(form.sttModel),
    tts_model: text(form.ttsModel),
    llm_model: text(form.llmModel),
    pipeline: text(form.pipeline),
    max_duration_message: text(form.maxDurationMessage),
    silence_prompt: form.silencePrompt.trim() || 'Are you still there?',
    goodbye: text(form.goodbye),
    transfer_number: text(form.transferNumber),
    transfer_message: text(form.transferMessage),
    recording_consent: text(form.recordingConsent),
    voicemail: form.voicemail || 'hangup',
    voicemail_message: text(form.voicemailMessage),
  }
  let problem: string | null = null
  const number = (raw: string, what: string): number => {
    const value = Number(raw)
    if (raw.trim() === '' || !Number.isFinite(value) || value < 0) problem ??= `${what} must be a number, 0 or more.`
    return value
  }
  body.max_duration_s = number(form.maxDurationS, 'The longest call length')
  body.silence_timeout_s = number(form.silenceTimeoutS, 'The silence timeout')
  body.silence_prompts = Math.round(number(form.silencePrompts, 'The number of prompts'))
  if (body.transfer_number && !/^\+?[0-9]{5,15}$|^sips?:\S+$/.test(body.transfer_number))
    problem ??= 'The transfer number must be in international form, for example +14155550100.'
  if (body.voicemail === 'leave_message' && !body.voicemail_message)
    problem ??= 'Leaving a voicemail needs the message to leave.'
  body.tools = []
  if (form.tools.trim()) {
    try {
      const parsed: unknown = JSON.parse(form.tools)
      if (
        !Array.isArray(parsed) ||
        !parsed.every((t) => t && typeof t === 'object' && typeof (t as { name?: unknown }).name === 'string')
      )
        problem ??= 'Tools must be a JSON list of objects, each with a name.'
      else body.tools = parsed as PhoneAgentUpdate['tools']
    } catch {
      problem ??= 'Tools are not valid JSON.'
    }
  }
  return { body, problem }
}
