import { HardDrive, Trash2, TriangleAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import { browserModel, BROWSER_MODELS } from '@/lib/browser-stt/models'
import { BROWSER_VOICES } from '@/lib/browser-tts/voices'
import {
  CAP_CHOICES,
  clearAllDownloads,
  deleteDownload,
  dismissInterrupted,
  downloads,
  interrupted,
  residentIds,
  setStorageCap,
  storageUse,
  totalBytes,
  type DownloadEntry,
  type StorageUse,
} from '@/lib/downloads'
import { fmtBytes } from '@/lib/format'
import { cn } from '@/lib/utils'

function ago(ms: number): string {
  const s = Math.max(0, (Date.now() - ms) / 1000)
  if (s < 90) return 'just now'
  if (s < 3600 * 1.5) return `${Math.round(s / 60)} min ago`
  if (s < 86400 * 1.5) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} days ago`
}

function modelLabel(id: string): string {
  return (
    BROWSER_VOICES.find((v) => v.id === id)?.label ??
    (BROWSER_MODELS.some((m) => m.id === id) ? browserModel(id).label : id)
  )
}

/** Re-read what the browser says this site stores whenever the list changes. */
function useStorageUse(): StorageUse | null {
  const state = useStore(downloads)
  const [use, setUse] = useState<StorageUse | null>(null)
  useEffect(() => {
    let live = true
    void storageUse().then((u) => live && setUse(u))
    return () => {
      live = false
    }
  }, [state.entries])
  return use
}

const capLabel = (c: number | null) => (c == null ? 'No limit' : fmtBytes(c))

/**
 * Every model this browser has downloaded for the in-page recogniser and
 * voices: size, when it was last used, delete one or all, and an optional
 * limit that removes the least recently used ones.
 */
const KIND_TEXT: Record<DownloadEntry['kind'], string> = {
  tts: 'Voice',
  stt: 'Speech to text',
  diarize: 'Speaker diarization',
  llm: 'Language model',
  clone: 'Voice cloning',
}

export function DownloadsPanel({ className }: { className?: string }) {
  const state = useStore(downloads)
  const use = useStorageUse()
  const [busy, setBusy] = useState(false)
  const entries = [...state.entries].sort((a, b) => b.lastUsed - a.lastUsed)
  const listed = totalBytes(entries)
  const resident = residentIds()
  // Anything the list does not account for: downloads from before it existed, the runtime, the page itself.
  const other = use?.usage != null ? Math.max(0, use.usage - listed) : 0

  const remove = async (e: DownloadEntry) => {
    setBusy(true)
    try {
      await deleteDownload(e.id)
      toast.success(`Removed ${e.label}`)
    } finally {
      setBusy(false)
    }
  }
  const clear = async () => {
    setBusy(true)
    try {
      await clearAllDownloads()
      toast.success('Removed every downloaded model from this browser')
    } finally {
      setBusy(false)
    }
  }
  const choose = (value: string) => {
    const cap = value === 'none' ? null : Number(value)
    void setStorageCap(cap).then((removed) => {
      if (removed.length)
        toast.message(
          `Removed ${removed.length} model${removed.length === 1 ? '' : 's'} to stay under ${capLabel(cap)}`,
        )
    })
  }

  return (
    <Panel
      title="Downloaded models"
      description={`${entries.length} model${entries.length === 1 ? '' : 's'} · ${fmtBytes(listed)}`}
      className={className}
      data-testid="downloads-panel"
      actions={
        <Button
          size="sm"
          variant="outline"
          onClick={() => void clear()}
          disabled={busy || (entries.length === 0 && other < 1e6)}
          data-testid="clear-downloads"
        >
          <Trash2 /> Clear all
        </Button>
      }
      bodyClassName="grid gap-3 text-[13px] leading-5"
    >
      {entries.length === 0 ? (
        <p className="text-muted-foreground">
          Nothing yet. Each model is downloaded the first time you use it and kept by this browser until you remove it.
        </p>
      ) : (
        <ul className="grid gap-1.5" aria-label="Downloaded models">
          {entries.map((e) => (
            <li
              key={e.id}
              className="flex min-w-0 items-center gap-2 rounded-lg border border-border px-2.5 py-1.5"
              data-testid={`download-${e.id}`}
            >
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{e.label}</div>
                <div className="truncate text-xs text-muted-foreground">
                  {KIND_TEXT[e.kind]} · <span className="num">{fmtBytes(e.bytes)}</span> · used {ago(e.lastUsed)}
                  {resident.includes(e.id) && ' · loaded now'}
                </div>
              </div>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label={`Delete ${e.label}`}
                onClick={() => void remove(e)}
                disabled={busy}
                data-testid={`delete-${e.id}`}
              >
                <Trash2 />
              </Button>
            </li>
          ))}
        </ul>
      )}
      {state.evicted.length > 0 && (
        <p className="text-muted-foreground" data-testid="evicted-note">
          Removed to stay under the limit: {state.evicted.map(modelLabel).join(', ')}.
        </p>
      )}
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <p className="min-w-0 text-xs text-muted-foreground" data-testid="storage-use">
          <HardDrive className="mr-1 inline size-3.5 align-[-2px]" aria-hidden />
          {use?.usage != null ? (
            <>
              This site stores <span className="num">{fmtBytes(use.usage)}</span>
              {use.quota != null && (
                <>
                  {' '}
                  of <span className="num">{fmtBytes(use.quota)}</span> the browser allows
                </>
              )}
              {other > 20e6 && (
                <>
                  {' '}
                  (<span className="num">{fmtBytes(other)}</span> of it is the runtime, the page and anything downloaded
                  before this list existed; Clear all removes the models among it)
                </>
              )}
              .
            </>
          ) : (
            'This browser does not say how much it stores.'
          )}
        </p>
        <label className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Storage limit</span>
          <Select value={state.capBytes == null ? 'none' : String(state.capBytes)} onValueChange={choose}>
            <SelectTrigger
              className="h-7 w-28"
              aria-label="Storage limit for downloaded models"
              data-testid="storage-cap"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CAP_CHOICES.map((c) => (
                <SelectItem key={String(c)} value={c == null ? 'none' : String(c)}>
                  {capLabel(c)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
      </div>
      <p className="text-xs text-muted-foreground">
        With a limit, the models used least recently are removed when a new download would go over it; a model that is
        loaded is never removed. Without one, the browser decides when it runs out of space.
      </p>
    </Panel>
  )
}

/**
 * Shown once after the page closed in the middle of loading a model (almost
 * always the browser reclaiming memory): nothing is loaded by itself this
 * time, and the downloads can be cleared.
 */
export function InterruptedLoadNotice({ className }: { className?: string }) {
  const last = useStore(interrupted)
  if (!last) return null
  return (
    <Alert className={cn('mb-4', className)} data-testid="interrupted-load">
      <TriangleAlert />
      <AlertTitle>The page closed while loading {modelLabel(last.id)}</AlertTitle>
      <AlertDescription>
        <p>
          That usually means the browser ran out of memory for this tab. Nothing loads by itself this time. Close other
          tabs, choose a smaller model, or clear the downloads and start afresh.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              void clearAllDownloads().then(() => {
                dismissInterrupted()
                toast.success('Removed every downloaded model from this browser')
              })
            }
            data-testid="interrupted-clear"
          >
            <Trash2 /> Clear downloads
          </Button>
          <Button size="sm" variant="ghost" onClick={dismissInterrupted} data-testid="interrupted-dismiss">
            Dismiss
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  )
}
