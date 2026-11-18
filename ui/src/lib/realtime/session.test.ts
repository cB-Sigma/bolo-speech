import { describe, expect, it } from 'vitest'
import { applyEvent, decodeAudioDelta, emptyConversation, InputEncoder, sessionUpdate } from './session'

describe('conversation events', () => {
  it('sends the pipeline in the first session.update', () => {
    const u = sessionUpdate('speech-to-speech') as {
      session: { pipeline: string; audio: { input: { format: { rate: number } } } }
    }
    expect(u.session.pipeline).toBe('speech-to-speech')
    expect(u.session.audio.input.format.rate).toBe(24000)
  })

  it('builds user and agent turns from deltas, completions and response.done', () => {
    let s = emptyConversation('cascade')
    const ev = (e: Record<string, unknown>) => (s = applyEvent(s, e as { type: string }))
    ev({ type: 'session.updated', session: { pipeline: 'audio-llm' } })
    ev({ type: 'input_audio_buffer.speech_started' })
    ev({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'i1', delta: 'hel' })
    ev({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'i1', delta: 'lo' })
    expect(s.userSpeaking).toBe(true)
    expect(s.turns).toEqual([{ id: 'i1', role: 'user', text: 'hello', final: false }])
    ev({ type: 'input_audio_buffer.speech_stopped' })
    ev({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'i1', transcript: 'Hello.' })
    ev({ type: 'response.output_audio_transcript.delta', response_id: 'r1', delta: 'Hi' })
    ev({ type: 'response.done', response: { id: 'r1', status: 'cancelled' } })
    expect(s.serverPipeline).toBe('audio-llm')
    expect(s.turns[0]).toMatchObject({ text: 'Hello.', final: true })
    expect(s.turns[1]).toMatchObject({ role: 'assistant', text: 'Hi', final: true, status: 'cancelled' })
    ev({ type: 'error', error: { message: 'pipeline cannot change', code: 'invalid_request' } })
    expect(s.error).toBe('pipeline cannot change')
    expect(s.events.at(-1)).toBe('error')
  })

  it('sends MCP tools and follows the servers, calls and approval requests', () => {
    const tools = [{ type: 'mcp', server_label: 'files' }]
    const u = sessionUpdate('cascade', undefined, 'websocket', undefined, undefined, { tools }) as {
      session: { tools?: unknown }
    }
    expect(u.session.tools).toEqual(tools)
    expect((sessionUpdate('cascade') as { session: { tools?: unknown } }).session.tools).toBeUndefined()

    let s = emptyConversation('cascade')
    const ev = (e: Record<string, unknown>) => (s = applyEvent(s, e as { type: string }))
    const list = { id: 'l1', type: 'mcp_list_tools', server_label: 'files', tools: [] }
    ev({ type: 'conversation.item.added', item: list })
    expect(s.mcpServers.files).toEqual({ status: 'listing', tools: [], error: null })
    ev({ type: 'conversation.item.done', item: { ...list, tools: [{ name: 'read' }, { name: 'list' }] } })
    expect(s.mcpServers.files).toEqual({ status: 'ready', tools: ['read', 'list'], error: null })
    ev({ type: 'conversation.item.done', item: { ...list, server_label: 'gone', error: 'refused' } })
    expect(s.mcpServers.gone?.status).toBe('failed')

    const call = { id: 'm1', type: 'mcp_call', server_label: 'files', name: 'read', arguments: '', output: null }
    ev({ type: 'conversation.item.added', item: call })
    expect(s.toolCalls[0]?.status).toBe('running')
    ev({ type: 'conversation.item.done', item: { ...call, arguments: '{"p":1}', output: 'contents' } })
    expect(s.toolCalls[0]).toMatchObject({ status: 'done', output: 'contents', arguments: '{"p":1}' })

    const req = { id: 'r1', type: 'mcp_approval_request', server_label: 'files', name: 'list', arguments: '{}' }
    ev({ type: 'conversation.item.added', item: req })
    ev({ type: 'conversation.item.done', item: req })
    expect(s.toolCalls.filter((c) => c.status === 'approval')).toHaveLength(1)
    ev({ type: 'conversation.item.added', item: { ...call, id: 'm2', approval_request_id: 'r1' } })
    expect(s.toolCalls.find((c) => c.id === 'r1')?.status).toBe('allowed')
    ev({
      type: 'conversation.item.done',
      item: { ...call, id: 'm2', approval_request_id: 'r1', error: { message: 'boom' } },
    })
    expect(s.toolCalls.find((c) => c.id === 'm2')).toMatchObject({ status: 'failed', output: 'boom' })
  })

  it('reports a failed response as an error', () => {
    const s = applyEvent(emptyConversation(), {
      type: 'response.done',
      response: { id: 'r', status: 'failed', status_details: { error: { message: 'model unreachable' } } },
    })
    expect(s.error).toBe('model unreachable')
  })
})

describe('audio framing', () => {
  it('encodes 16 kHz blocks as 24 kHz PCM16 base64, which decodes back', () => {
    const enc = new InputEncoder()
    const block = new Float32Array(320).fill(0.5)
    enc.encode(block)
    const out = decodeAudioDelta(enc.encode(block))
    expect(out.length).toBe(480)
    expect(out[240]).toBeCloseTo(0.5, 2)
  })
})
