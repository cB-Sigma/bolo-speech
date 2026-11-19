import { BookOpen, Menu, Monitor, Moon, Search, Sun } from 'lucide-react'
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Logo } from '@/components/logo'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useInfo, useLive, useNavFeatures } from '@/hooks/use-api'
import { useTheme, type ThemeChoice } from '@/hooks/use-theme'
import { DemoBanner } from '@/components/demo'
import { isDemo, isMock } from '@/lib/api/config'
import { isDebug } from '@/lib/diagnostics'
import { fmtDuration } from '@/lib/format'
import { NAV, visibleNav, type NavItem } from '@/lib/nav'
import { cn } from '@/lib/utils'
import { openCommandPalette, PaletteHost } from './palette-host'
import { ApiKeyDialog, ConnectionIndicator, OfflineBanner } from './connection'
import { DownloadTray } from './download-tray'

const DiagnosticsPanel = lazy(() =>
  import('@/components/diagnostics-panel').then((m) => ({ default: m.DiagnosticsPanel })),
)

export const DOCS_URL = 'https://github.com/cB-Sigma/bolo-speech/tree/main/docs'

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  const groups = ['Playground', 'Server'] as const
  const items = visibleNav(useNavFeatures())
  return (
    <nav aria-label="Main" className="flex flex-col gap-5">
      {groups.map((g) => (
        <div key={g}>
          <div className="mb-1.5 px-2.5 eyebrow">{g}</div>
          <ul className="flex flex-col gap-0.5">
            {items
              .filter((n) => n.group === g)
              .map((n) => (
                <li key={n.to}>
                  <NavItemLink item={n} onNavigate={onNavigate} />
                </li>
              ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function NavItemLink({ item, onNavigate }: { item: NavItem; onNavigate?: () => void }) {
  return (
    <NavLink
      to={item.to}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          'group flex h-8 items-center gap-2.5 rounded-md px-2.5 text-sm transition-colors duration-150',
          isActive
            ? 'bg-sidebar-accent font-medium text-foreground'
            : 'text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground',
        )
      }
    >
      {({ isActive }) => (
        <>
          <item.icon className={cn('size-4 shrink-0', isActive ? 'text-link' : '')} aria-hidden />
          <span className="truncate">{item.label}</span>
        </>
      )}
    </NavLink>
  )
}

function ServerFooter() {
  const info = useInfo()
  const feed = useLive()
  return (
    <div className="mt-auto border-t border-sidebar-border px-4 py-3 text-xs text-muted-foreground">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate num text-foreground">{info.data ? `bolo ${info.data.version}` : 'bolo'}</span>
        <span className="num">{info.data?.device ?? ''}</span>
      </div>
      <div className="mt-1 flex items-center justify-between gap-2">
        <span>Uptime</span>
        <span className="num">{feed.stats ? fmtDuration(feed.stats.uptime_s) : '–'}</span>
      </div>
      <a
        href={DOCS_URL}
        target="_blank"
        rel="noreferrer"
        className="mt-2 inline-flex items-center gap-1.5 rounded-sm text-muted-foreground hover:text-foreground"
      >
        <BookOpen className="size-3.5" aria-hidden /> Documentation
      </a>
    </div>
  )
}

function ThemeToggle() {
  const { choice, resolved, setChoice } = useTheme()
  const Icon = choice === 'system' ? Monitor : resolved === 'dark' ? Moon : Sun
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={`Theme: ${choice}`}>
              <Icon />
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom">Theme</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="min-w-36">
        <DropdownMenuRadioGroup value={choice} onValueChange={(v) => setChoice(v as ThemeChoice)}>
          <DropdownMenuRadioItem value="light">
            <Sun /> Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark">
            <Moon /> Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system">
            <Monitor /> System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** `g` then a digit jumps to the nth screen, as listed in the command palette. */
function useGoShortcuts() {
  const navigate = useNavigate()
  const armed = useRef(0)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return
      if (e.key === 'g') {
        armed.current = Date.now()
        return
      }
      if (Date.now() - armed.current < 1200 && /^[1-9]$/.test(e.key)) {
        const item = NAV[Number(e.key) - 1]
        if (item) navigate(item.to)
      }
      armed.current = 0
    }
    globalThis.addEventListener('keydown', onKey)
    return () => globalThis.removeEventListener('keydown', onKey)
  }, [navigate])
}

export function AppShell() {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const current = NAV.find((n) => location.pathname.startsWith(n.to))
  const mainRef = useRef<HTMLElement>(null)
  useGoShortcuts()

  useEffect(() => {
    document.title = current ? `${current.label} · bolo` : 'bolo'
  }, [current])

  const debug = useMemo(() => isDebug(), [])
  const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

  return (
    <div className="flex min-h-svh bg-background pr-[env(safe-area-inset-right)] pl-[env(safe-area-inset-left)]">
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault()
          mainRef.current?.focus()
        }}
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-sm focus:text-primary-foreground"
      >
        Skip to content
      </a>

      <aside className="sticky top-0 hidden h-svh w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar lg:flex">
        <div className="flex h-14 items-center px-4">
          <NavLink to="/stt" aria-label="bolo home" className="rounded-md">
            <Logo />
          </NavLink>
        </div>
        <div className="flex-1 overflow-y-auto px-2 pt-2">
          <NavList />
        </div>
        <ServerFooter />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-[calc(3.5rem+env(safe-area-inset-top))] items-center gap-2 border-b border-border bg-background/85 px-3 pt-[env(safe-area-inset-top)] backdrop-blur-md sm:px-5">
          <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open navigation">
                <Menu />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 gap-0 bg-sidebar p-0">
              <SheetHeader className="h-14 justify-center px-4">
                <SheetTitle>
                  <Logo />
                </SheetTitle>
                <SheetDescription className="sr-only">Navigation</SheetDescription>
              </SheetHeader>
              <div className="flex-1 overflow-y-auto px-2 pt-2">
                <NavList onNavigate={() => setMenuOpen(false)} />
              </div>
              <ServerFooter />
            </SheetContent>
          </Sheet>
          <NavLink to="/stt" aria-label="bolo home" className="rounded-md lg:hidden">
            <Logo hindi="wide" />
          </NavLink>
          <div className="hidden min-w-0 items-baseline gap-2.5 lg:flex">
            <h1 className="truncate text-sm font-semibold">{current?.label ?? 'bolo'}</h1>
            <span className="truncate text-sm text-muted-foreground">{current?.description}</span>
          </div>
          <div className="ml-auto flex items-center gap-1">
            {isMock && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="mr-1 hidden cursor-default sm:inline-flex" tabIndex={0}>
                    Mock data
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  Simulated server. Numbers are illustrative, not measurements.
                </TooltipContent>
              </Tooltip>
            )}
            <ConnectionIndicator />
            <Button
              variant="outline"
              size="sm"
              onClick={openCommandPalette}
              className="hidden h-8 w-44 justify-start gap-2 px-2.5 font-normal text-muted-foreground md:inline-flex"
            >
              <Search className="size-3.5" />
              <span>Search</span>
              <kbd className="ml-auto rounded border border-border bg-muted px-1.5 num text-[10px] leading-4">
                {isMac ? '⌘K' : 'Ctrl K'}
              </kbd>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={openCommandPalette}
              className="md:hidden"
              aria-label="Open command palette"
            >
              <Search />
            </Button>
            <ThemeToggle />
          </div>
        </header>
        {isDemo && <DemoBanner />}
        <OfflineBanner />
        <main
          id="main"
          ref={mainRef}
          tabIndex={-1}
          className="min-w-0 flex-1 pb-[env(safe-area-inset-bottom)] outline-none"
        >
          <Outlet />
          {debug && (
            <Suspense fallback={null}>
              <DiagnosticsPanel />
            </Suspense>
          )}
        </main>
      </div>
      <PaletteHost />
      <ApiKeyDialog />
      <DownloadTray />
    </div>
  )
}

/** Standard page container: a mobile-only title, optional toolbar, then content. */
export function Page({
  title,
  toolbar,
  children,
  className,
}: {
  title: string
  toolbar?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('mx-auto w-full max-w-[1400px] px-4 py-4 sm:px-6 sm:py-6', className)}>
      <h1 className="mb-3 text-lg font-semibold tracking-tight lg:hidden">{title}</h1>
      {toolbar && <div className="mb-4 flex min-w-0 flex-wrap items-center gap-2">{toolbar}</div>}
      {children}
    </div>
  )
}
