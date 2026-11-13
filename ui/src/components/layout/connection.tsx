import { useQueryClient } from '@tanstack/react-query'
import { KeyRound, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useInfo, useLive } from '@/hooks/use-api'
import { NetworkError } from '@/lib/api/client'
import { getApiKey, isDemo, serverBase, setApiKey } from '@/lib/api/config'
import { live } from '@/lib/live'
import { cn } from '@/lib/utils'

export type ConnState = 'online' | 'connecting' | 'reconnecting' | 'offline' | 'locked'

export function useConnection(): { state: ConnState; label: string; detail: string } {
  const info = useInfo()
  const feed = useLive()
  const offline = info.error instanceof NetworkError || (info.isError && !info.data)
  if (feed.status === 'unauthorized')
    return { state: 'locked', label: 'API key needed', detail: 'This server requires an API key.' }
  if (offline && feed.status !== 'open') {
    return { state: 'offline', label: 'Offline', detail: `Cannot reach ${serverBase()}. Retrying…` }
  }
  if (feed.status === 'open') {
    return isDemo
      ? { state: 'online', label: 'In browser', detail: 'No server: everything runs in this page.' }
      : { state: 'online', label: 'Live', detail: `Connected to ${serverBase()}` }
  }
  if (feed.status === 'reconnecting')
    return { state: 'reconnecting', label: 'Reconnecting', detail: 'The live feed dropped; retrying.' }
  return { state: 'connecting', label: 'Connecting', detail: `Connecting to ${serverBase()}` }
}

const DOT: Record<ConnState, string> = {
  online: 'bg-good',
  connecting: 'bg-warning live-dot',
  reconnecting: 'bg-warning live-dot',
  offline: 'bg-critical',
  locked: 'bg-warning',
}

export function ConnectionIndicator({ className }: { className?: string }) {
  const { state, label, detail } = useConnection()
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          role="status"
          aria-live="polite"
          tabIndex={0}
          className={cn('inline-flex h-7 items-center gap-2 rounded-md px-2 text-xs text-muted-foreground', className)}
          data-state={state}
          data-testid="connection"
        >
          <span className={cn('size-2 rounded-full', DOT[state])} aria-hidden />
          <span className="font-medium text-foreground">{label}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom">{detail}</TooltipContent>
    </Tooltip>
  )
}

/** Full-width notice when the server cannot be reached. */
export function OfflineBanner() {
  const { state } = useConnection()
  const qc = useQueryClient()
  if (state !== 'offline') return null
  return (
    <div
      role="alert"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-critical/30 bg-critical/10 px-4 py-2 text-sm"
    >
      <span className="size-2 rounded-full bg-critical" aria-hidden />
      <span className="font-medium">Server unreachable.</span>
      <span className="text-muted-foreground">
        Is <code className="num text-foreground">bolo serve</code> running? Retrying automatically.
      </span>
      <Button
        size="xs"
        variant="outline"
        className="ml-auto"
        onClick={() => {
          void qc.invalidateQueries()
          live.restart()
        }}
      >
        <RefreshCw /> Retry now
      </Button>
    </div>
  )
}

/** Prompts for an API key when the server demands one. */
export function ApiKeyDialog() {
  const info = useInfo()
  const feed = useLive()
  const qc = useQueryClient()
  // Opens by itself when a key is needed, until dismissed; can also be opened on request.
  const [requested, setRequested] = useState(false)
  const [dismissed, setDismissed] = useState(false)
  const [value, setValue] = useState('')
  const needed = (info.data?.auth_required && !getApiKey()) || feed.status === 'unauthorized'
  const open = requested || (Boolean(needed) && !dismissed)
  const setOpen = (o: boolean) => {
    setRequested(false)
    setDismissed(!o)
  }

  useEffect(() => {
    const on = () => setRequested(true)
    globalThis.addEventListener('bolo:ask-key', on)
    return () => globalThis.removeEventListener('bolo:ask-key', on)
  }, [])

  const save = () => {
    setApiKey(value.trim() || null)
    // Close for now; if the server rejects the key the dialog comes back by itself.
    setRequested(false)
    setDismissed(false)
    setValue('')
    live.restart()
    void qc.invalidateQueries()
    toast.success(value.trim() ? 'API key saved in this browser' : 'API key cleared')
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="size-4" /> API key
          </DialogTitle>
          <DialogDescription>
            {feed.status === 'unauthorized' && getApiKey()
              ? 'The server rejected the saved key. Enter a valid one.'
              : 'This server requires an API key. It is stored only in this browser and sent as a Bearer token.'}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            save()
          }}
          className="grid gap-2"
        >
          <Label htmlFor="api-key">Key</Label>
          <Input
            id="api-key"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="sk-…"
            className="num"
            autoFocus
          />
          <DialogFooter className="mt-2">
            <Button type="button" variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button type="submit">Save key</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function askForApiKey(): void {
  globalThis.dispatchEvent(new Event('bolo:ask-key'))
}
