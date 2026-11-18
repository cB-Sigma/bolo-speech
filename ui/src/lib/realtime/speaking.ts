import type { RealtimeTransport } from './session'

/**
 * Whether the agent is audibly speaking: the Conversation page's "Agent is speaking".
 *
 * The answer's audio is generated faster than it is played, so `response.done` (generation finished) comes seconds
 * before the voice falls silent, and says nothing about playback. Where the playback happens decides who knows:
 *
 * - WebRTC: the server plays the answer on the call's audio track and says what the track does
 *   (`output_audio_buffer.started`, `.stopped`, `.cleared`); those events drive the indicator.
 * - WebSocket: this page plays the `response.output_audio.delta` chunks itself, and the server cannot know when they
 *   end. The page's own playback queue drives it: on when the first queued audio starts playing (`played`), off when
 *   the queue has run dry (`drained`) or the page cut it short (`interrupted`: barge-in, Stop the answer).
 *
 * On either, `input_audio_buffer.speech_started` turns it off at once (barge-in stops the playback), except with a
 * full-duplex model, which goes on talking over the user.
 */
export type SpeakingInput =
  { kind: 'event'; type: string } | { kind: 'played' } | { kind: 'drained' } | { kind: 'interrupted' }

export function nextSpeaking(
  speaking: boolean,
  input: SpeakingInput,
  transport: RealtimeTransport,
  fullDuplex = false,
): boolean {
  if (input.kind === 'interrupted') return false
  if (input.kind === 'played' || input.kind === 'drained') {
    // the page's own playback: what counts over WebSocket (over WebRTC the page plays nothing itself)
    return transport === 'websocket' ? input.kind === 'played' : speaking
  }
  switch (input.type) {
    case 'input_audio_buffer.speech_started':
      return fullDuplex ? speaking : false
    case 'output_audio_buffer.started':
      return transport === 'webrtc' ? true : speaking
    case 'output_audio_buffer.stopped':
    case 'output_audio_buffer.cleared':
      return transport === 'webrtc' ? false : speaking
    default:
      // response.done, response.output_audio.done and the rest: generation, not playback
      return speaking
  }
}

/** How long the playback queue must stay empty before the agent counts as quiet: the gap between two sentences
 * whose audio arrived a little late is not the end of the answer. */
export const DRAIN_GRACE_MS = 250
