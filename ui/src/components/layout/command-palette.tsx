import { Check, KeyRound, Monitor, Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from '@/components/ui/command'
import { useCatalog, useWordLists } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { useTheme } from '@/hooks/use-theme'
import { visibleNav } from '@/lib/nav'
import { prefs, setPref } from '@/lib/prefs'
import { askForApiKey } from './connection'

/** Loaded on first use (see palette-host.tsx); opens immediately when mounted. */
export default function CommandPalette() {
  const [open, setOpen] = useState(true)
  const navigate = useNavigate()
  const { setChoice } = useTheme()
  const catalog = useCatalog()
  const p = useStore(prefs)
  const nav = visibleNav({ wordLists: useWordLists().offered })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    const onOpen = () => setOpen(true)
    globalThis.addEventListener('keydown', onKey)
    globalThis.addEventListener('bolo:palette', onOpen)
    return () => {
      globalThis.removeEventListener('keydown', onKey)
      globalThis.removeEventListener('bolo:palette', onOpen)
    }
  }, [])

  const run = (fn: () => void) => () => {
    setOpen(false)
    fn()
  }
  const usable = (catalog.data?.models ?? []).filter((m) => m.installed && (m.task === 'stt' || m.task === 'tts'))

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      description="Jump to a screen or switch model"
    >
      <Command>
        <CommandInput placeholder="Go to a screen, switch model…" />
        <CommandList>
          <CommandEmpty>Nothing matches.</CommandEmpty>
          <CommandGroup heading="Go to">
            {nav.map((n, i) => (
              <CommandItem
                key={n.to}
                value={n.label}
                keywords={['go', n.description]}
                onSelect={run(() => navigate(n.to))}
              >
                <n.icon />
                <span>{n.label}</span>
                <CommandShortcut>G {i + 1}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
          {usable.length > 0 && (
            <>
              <CommandGroup heading="Use model">
                {usable.map((m) => {
                  const active = (m.task === 'stt' ? p.sttModel : p.ttsModel) === m.id
                  return (
                    <CommandItem
                      key={m.id}
                      value={m.id}
                      keywords={['model', m.family, m.task]}
                      onSelect={run(() => {
                        setPref(m.task === 'stt' ? 'sttModel' : 'ttsModel', m.id)
                        navigate(m.task === 'stt' ? '/stt' : '/tts')
                      })}
                    >
                      {active ? <Check /> : <span className="size-4" />}
                      <span className="num text-[13px]">{m.id}</span>
                      <span className="ml-auto text-xs text-muted-foreground">
                        {m.task === 'stt' ? 'Speech to text' : 'Text to speech'} · {m.tier}
                      </span>
                    </CommandItem>
                  )
                })}
              </CommandGroup>
            </>
          )}
          <CommandGroup heading="Preferences">
            <CommandItem value="theme light" onSelect={run(() => setChoice('light'))}>
              <Sun /> Light theme
            </CommandItem>
            <CommandItem value="theme dark" onSelect={run(() => setChoice('dark'))}>
              <Moon /> Dark theme
            </CommandItem>
            <CommandItem value="theme system" onSelect={run(() => setChoice('system'))}>
              <Monitor /> Follow system theme
            </CommandItem>
            <CommandItem value="api key authentication" onSelect={run(askForApiKey)}>
              <KeyRound /> Set API key
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
