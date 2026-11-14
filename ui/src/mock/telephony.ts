import type {
  CallRecord,
  CallSummary,
  PhoneAgent,
  PhoneAgentUpdate,
  PhoneNumber,
  SimulationReport,
  TelephonyOverview,
  TelephonyProvider,
} from '@/lib/api/types'

const BASE = 'https://voice.example.org'

function provider(name: string, label: string, over: Partial<TelephonyProvider>): TelephonyProvider {
  return {
    name,
    label,
    configured: false,
    exposed: false,
    problems: [],
    protection: 'signature',
    can_validate: false,
    can_place_calls: false,
    machine_detection: true,
    ends_by: 'close',
    settings: {},
    docs_url: `https://example.org/${name}`,
    webhooks: [
      { name: 'Answer webhook', where: `${label} dashboard`, url: `${BASE}/telephony/${name}/voice`, method: 'POST' },
    ],
    stream_url: `wss://voice.example.org/telephony/${name}/stream/<token>`,
    notes: [],
    ...over,
  }
}

export function blankAgent(name: string): PhoneAgent {
  return {
    name,
    instructions: null,
    greeting: null,
    voice: null,
    language: null,
    stt_model: null,
    tts_model: null,
    llm_model: null,
    pipeline: null,
    tools: [],
    max_duration_s: 600,
    max_duration_message: null,
    silence_timeout_s: 10,
    silence_prompt: 'Are you still there?',
    silence_prompts: 2,
    goodbye: null,
    transfer_number: null,
    transfer_message: null,
    recording_consent: null,
    voicemail: 'hangup',
    voicemail_message: null,
    dtmf: 'message',
    summary: true,
    turn_detection: null,
    source: 'stored',
    editable: true,
  }
}

function sampleCall(id: string, over: Partial<CallRecord> = {}): CallRecord {
  const q = (v: number) => ({ p50: v, max: v + 120, mean: v + 30 })
  return {
    id,
    provider: 'twilio',
    direction: 'inbound',
    agent: 'front-desk',
    from: '+14155550123',
    to: '+14155550100',
    provider_call_id: 'CA0123456789',
    created_at: '2026-11-14T09:30:00.000Z',
    started_at: '2026-11-14T09:30:01.000Z',
    ended_at: '2026-11-14T09:30:19.000Z',
    status: 'ended',
    outcome: 'caller_hangup',
    error: null,
    answered_by: null,
    duration_s: 18.2,
    pipeline: 'cascade',
    interruptions: 1,
    latency: {
      turns: 2,
      endpoint_ms: q(500),
      stt_ms: q(90),
      llm_first_token_ms: q(310),
      tts_first_audio_ms: q(160),
      send_ms: q(1),
      total_ms: q(561),
      response_ms: q(1061),
    },
    summary: 'A caller asked about opening hours and was told the desk is open until six.',
    stream_id: 'MZ0123',
    transfer_to: null,
    dtmf: '',
    summary_error: null,
    metadata: {},
    timeline: [
      { t_ms: 0, type: 'ringing' },
      { t_ms: 0, type: 'stream_started', encoding: 'mulaw', sample_rate: 8000 },
      { t_ms: 4, type: 'greeting', text: 'Hello, front desk.' },
      { t_ms: 2400, type: 'user_speech_started' },
      { t_ms: 4900, type: 'user_speech_stopped' },
      { t_ms: 5461, type: 'agent_audio_started', turn: 0, response_ms: 1061 },
      { t_ms: 7900, type: 'barge_in', cause: 'speech', heard_ms: 2400, cleared_ms: 1800 },
      { t_ms: 18200, type: 'ended', outcome: 'caller_hangup' },
    ],
    turns: [
      {
        index: 0,
        kind: 'speech',
        started_ms: 4900,
        user_text: 'What time do you close today?',
        assistant_text: 'We are open until six this evening.',
        interrupted: true,
        status: 'completed',
        tool_calls: [],
        latency: {
          endpoint_ms: 500,
          stt_ms: 90,
          llm_first_token_ms: 310,
          tts_first_audio_ms: 160,
          send_ms: 1,
          total_ms: 561,
          response_ms: 1061,
        },
      },
    ],
    transcript: [
      { t_ms: 4, speaker: 'agent', text: 'Hello, front desk.', kind: 'greeting' },
      { t_ms: 4990, speaker: 'caller', text: 'What time do you close today?' },
      {
        t_ms: 5600,
        speaker: 'agent',
        text: 'We are open until six this evening.',
        interrupted: true,
        full_text: 'We are open until six this evening. Is there anything else I can help you with?',
      },
      { t_ms: 9800, speaker: 'caller', text: 'Thanks, that is all.' },
    ],
    ...over,
  }
}

/** The telephony side of the mock server: one provider set up, one agent from the configuration, one call. */
export class MockTelephony {
  agents: PhoneAgent[] = [
    {
      ...blankAgent('front-desk'),
      instructions: 'You answer the phone for the front desk.',
      greeting: 'Hello, front desk.',
      source: 'settings',
      editable: false,
    },
  ]
  numbers: PhoneNumber[] = [{ number: '+14155550100', agent: 'front-desk', source: 'settings', editable: false }]
  calls: CallRecord[] = [sampleCall('call_000001_00001')]
  private seq = 1

  overview(): TelephonyOverview {
    return {
      public_base_url: BASE,
      validate_signatures: true,
      default_agent: 'front-desk',
      providers: [
        provider('twilio', 'Twilio', {
          configured: true,
          exposed: true,
          can_validate: true,
          can_place_calls: true,
          settings: { account_sid: 'AC0123', auth_token: '***', from_number: '+14155550100' },
          webhooks: [
            {
              name: 'A call comes in',
              where: "Phone Numbers > Active numbers > Voice Configuration: 'A call comes in' = Webhook, HTTP POST",
              url: `${BASE}/telephony/twilio/voice`,
              method: 'POST',
            },
          ],
          notes: ['bolo answers with TwiML <Connect><Stream>.'],
        }),
        provider('plivo', 'Plivo', { problems: ['telephony.providers.plivo.auth_token is not set'] }),
        provider('telnyx', 'Telnyx', {
          ends_by: 'api',
          problems: ['telephony.providers.telnyx.public_key is not set'],
        }),
        provider('exotel', 'Exotel', {
          protection: 'basic',
          machine_detection: false,
          problems: ['telephony.providers.exotel.stream_username is not set'],
        }),
      ],
      agents: this.agents,
      numbers: this.numbers,
      editable: true,
      edit_blocked_reason: null,
      llm_configured: true,
      stats: {
        active_calls: 0,
        calls_total: this.calls.length,
        by_outcome: [{ provider: 'twilio', outcome: 'caller_hangup', calls: 1 }],
      },
      compliance_notice:
        "Outbound calling is regulated. Consent, calling hours, do-not-call registers (for example TRAI's DND registry in India) and the disclosure that a machine is speaking are the operator's responsibility.",
    }
  }

  putAgent(name: string, body: PhoneAgentUpdate): PhoneAgent {
    const existing = this.agents.find((a) => a.name === name)
    if (existing && !existing.editable) throw new Error(`agent '${name}' comes from the configuration; change it there`)
    const next: PhoneAgent = { ...blankAgent(name), ...body, name, source: 'stored', editable: true }
    this.agents = existing ? this.agents.map((a) => (a.name === name ? next : a)) : [...this.agents, next]
    return next
  }

  deleteAgent(name: string): void {
    this.agents = this.agents.filter((a) => a.name !== name || !a.editable)
  }

  putNumber(number: string, agent: string): { numbers: PhoneNumber[] } {
    if (!this.agents.some((a) => a.name === agent)) throw new Error(`unknown agent '${agent}'`)
    this.numbers = [
      ...this.numbers.filter((n) => n.number !== number),
      { number, agent, source: 'stored', editable: true },
    ]
    return { numbers: this.numbers }
  }

  deleteNumber(number: string): void {
    this.numbers = this.numbers.filter((n) => n.number !== number || !n.editable)
  }

  summaries(): CallSummary[] {
    return this.calls.map(({ turns, ...rest }) => ({ ...rest, turns: turns.length }))
  }

  simulate(body: { agent?: string; provider?: string; scenario?: string }): SimulationReport {
    this.seq += 1
    const record = sampleCall(`call_00000${this.seq}_00001`, {
      provider: body.provider ?? 'twilio',
      agent: body.agent ?? 'front-desk',
      metadata: { simulated: true },
      summary: null,
    })
    this.calls = [record, ...this.calls]
    return {
      scenario: body.scenario ?? 'basic',
      description: 'The caller waits for the greeting, says one thing, hears the answer and hangs up.',
      provider: record.provider,
      agent: record.agent,
      call_id: record.id,
      passed: true,
      checks: [
        { type: 'agent_spoke', ok: true, detail: '6.2 s of agent audio heard' },
        { type: 'turns_min', ok: true, detail: '1 answered turn(s)' },
      ],
      duration_s: 10.3,
      noise_db: null,
      caller: {
        utterances: [{ label: 'speech', end_s: 4.2, seconds: 2, response_ms: 1090 }],
        response_ms: { p50: 1090, max: 1090, mean: 1090 },
        agent_audio_s: 6.2,
        clears: 0,
        cleared_s: 0,
      },
      outcome: 'caller_hangup',
      latency: record.latency,
      record,
    }
  }
}
