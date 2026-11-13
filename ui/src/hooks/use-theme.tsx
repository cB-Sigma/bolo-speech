import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type ThemeChoice = 'light' | 'dark' | 'system'

interface ThemeContextValue {
  choice: ThemeChoice
  resolved: 'light' | 'dark'
  setChoice: (c: ThemeChoice) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)
const KEY = 'bolo.theme'

function systemDark(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches
}

function readChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(readChoice)
  const [sysDark, setSysDark] = useState(systemDark)

  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const on = () => setSysDark(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  const resolved: 'light' | 'dark' = choice === 'system' ? (sysDark ? 'dark' : 'light') : choice

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', resolved === 'dark')
    root.dataset.theme = resolved
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', resolved === 'dark' ? '#0d0d0d' : '#f6f6f4')
  }, [resolved])

  const setChoice = useCallback((c: ThemeChoice) => {
    setChoiceState(c)
    try {
      if (c === 'system') localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, c)
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  }, [])

  const value = useMemo(() => ({ choice, resolved, setChoice }), [choice, resolved, setChoice])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>')
  return ctx
}
