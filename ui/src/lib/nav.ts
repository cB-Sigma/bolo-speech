import {
  Activity,
  AudioLines,
  AudioWaveform,
  BookA,
  Boxes,
  Cloud,
  Columns3,
  Gauge,
  MessagesSquare,
  Mic,
  Phone,
  Plug,
  Puzzle,
  Users,
  type LucideIcon,
} from 'lucide-react'
import { isDemo } from '@/lib/api/config'

/** Server features some screens depend on. */
export interface NavFeatures {
  wordLists: boolean
  /** Cloud providers behind `engine: remote` models. */
  providers?: boolean
  /** Phone calls through telephony providers. */
  telephony?: boolean
}

export interface NavItem {
  to: string
  label: string
  short: string
  description: string
  icon: LucideIcon
  group: 'Playground' | 'Server'
  /** Shown only when this returns true for the server's info (features newer servers add). */
  when?: (features: NavFeatures) => boolean
}

export const NAV: NavItem[] = [
  {
    to: '/stt',
    label: 'Speech to text',
    short: 'Transcribe',
    description: 'Transcribe live audio and inspect latency',
    icon: Mic,
    group: 'Playground',
  },
  {
    to: '/tts',
    label: 'Text to speech',
    short: 'Speak',
    description: 'Synthesise speech and watch it stream',
    icon: AudioLines,
    group: 'Playground',
  },
  {
    to: '/models',
    label: 'Models',
    short: 'Models',
    description: 'Browse, pull and load models',
    icon: Boxes,
    group: 'Server',
  },
  {
    to: '/latency',
    label: 'Latency',
    short: 'Latency',
    description: 'Live latency dashboard and request waterfall',
    icon: Activity,
    group: 'Server',
  },
  {
    to: '/benchmark',
    label: 'Benchmark',
    short: 'Bench',
    description: 'Sweep concurrency and find the ceiling',
    icon: Gauge,
    group: 'Server',
  },
  {
    to: '/connect',
    label: 'Connect',
    short: 'Connect',
    description: 'Client snippets and server details',
    icon: Plug,
    group: 'Server',
  },
  // Newer screens go last so the existing `g` + digit shortcuts keep their numbers.
  {
    to: '/conversation',
    label: 'Conversation',
    short: 'Talk',
    description: 'Talk to a voice agent and pick its pipeline',
    icon: MessagesSquare,
    group: 'Playground',
  },
  {
    to: '/components',
    label: 'Components',
    short: 'Components',
    description: 'Optional software some models need',
    icon: Puzzle,
    group: 'Server',
  },
  {
    to: '/wordlists',
    label: 'Word lists',
    short: 'Words',
    description: 'How voices pronounce names and terms',
    icon: BookA,
    group: 'Playground',
    when: (f) => f.wordLists,
  },
  {
    to: '/compare',
    label: 'Compare',
    short: 'Compare',
    description: 'Send the same audio or text to several models, local and cloud',
    icon: Columns3,
    group: 'Playground',
  },
  {
    to: '/providers',
    label: 'Providers',
    short: 'Providers',
    description: 'Cloud speech APIs: accounts, keys and cost',
    icon: Cloud,
    group: 'Server',
    when: (f) => Boolean(f.providers),
  },
  {
    to: '/speakers',
    label: 'Speakers',
    short: 'Speakers',
    description: 'Who spoke when, in this browser: examples, your audio, a transcript per speaker',
    icon: Users,
    group: 'Playground',
    when: () => isDemo,
  },
  {
    to: '/clone',
    label: 'Clone a voice',
    short: 'Clone',
    description: 'Copy a voice from a short recording, in this browser: nothing is uploaded',
    icon: AudioWaveform,
    group: 'Playground',
    when: () => isDemo,
  },
  {
    to: '/phone',
    label: 'Phone',
    short: 'Phone',
    description: 'Phone calls: providers, agents, numbers and recent calls',
    icon: Phone,
    group: 'Server',
    when: (f) => Boolean(f.telephony),
  },
]

/** The screens this server offers, in NAV order. */
export function visibleNav(features: NavFeatures): NavItem[] {
  return NAV.filter((n) => !n.when || n.when(features))
}
