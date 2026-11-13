import { lazy, Suspense, useEffect, useState } from 'react'

const CommandPalette = lazy(() => import('./command-palette'))

export function openCommandPalette(): void {
  globalThis.dispatchEvent(new Event('bolo:palette'))
}

/**
 * Keeps the command palette (and the search library behind it) out of the
 * main bundle: it is fetched the first time Ctrl/Cmd-K or the search button is used.
 */
export function PaletteHost() {
  const [wanted, setWanted] = useState(false)
  useEffect(() => {
    if (wanted) return // the palette handles its own shortcuts once mounted
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setWanted(true)
      }
    }
    const onOpen = () => setWanted(true)
    globalThis.addEventListener('keydown', onKey)
    globalThis.addEventListener('bolo:palette', onOpen)
    return () => {
      globalThis.removeEventListener('keydown', onKey)
      globalThis.removeEventListener('bolo:palette', onOpen)
    }
  }, [wanted])
  if (!wanted) return null
  return (
    <Suspense fallback={null}>
      <CommandPalette />
    </Suspense>
  )
}
