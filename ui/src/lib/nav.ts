import { Activity, AudioLines, Boxes, Gauge, Mic, Plug, type LucideIcon } from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  short: string
  description: string
  icon: LucideIcon
  group: 'Playground' | 'Server'
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
]
