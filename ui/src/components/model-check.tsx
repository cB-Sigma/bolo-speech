import { ClipboardCheck } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { ModelCheck } from '@/lib/api/types'
import { cn } from '@/lib/utils'

const STATUS_CLASS: Record<string, string> = {
  ok: 'text-good-text',
  fail: 'text-critical-text',
  warn: 'text-warning-text',
  skip: 'text-muted-foreground',
}

/**
 * "Check": would this model run here, and on bigger hardware? The server checks its runtime, every weight file at
 * its pinned revision and the placement on this machine, without downloading the weights (`bolo check`);
 * "Build it tiny" also loads a small random-weight copy of its architecture the way bolo would.
 */
export function ModelCheckButton({ id, className }: { id: string; className?: string }) {
  const [open, setOpen] = useState(false)
  const [tiny, setTiny] = useState(false)
  const [running, setRunning] = useState(false)
  const [report, setReport] = useState<ModelCheck | null>(null)
  const [error, setError] = useState<string | null>(null)
  const run = async () => {
    setRunning(true)
    setError(null)
    setReport(null)
    try {
      setReport(await api.checkModel(id, { tiny }))
    } catch (err) {
      setError(describeError(err))
    } finally {
      setRunning(false)
    }
  }
  return (
    <>
      <Button
        size="sm"
        variant="ghost"
        className={className}
        onClick={() => setOpen(true)}
        data-testid={`check-${id}`}
        title="Would it run here? Checked without downloading the weights"
      >
        <ClipboardCheck /> Check
      </Button>
      <Dialog open={open} onOpenChange={(o) => !running && setOpen(o)}>
        <DialogContent className="sm:max-w-xl" data-testid="check-dialog">
          <DialogHeader>
            <DialogTitle>Check {id}</DialogTitle>
            <DialogDescription>
              Its runtime, every weight file at its pinned revision and where it would run on this machine, without
              downloading the weights.
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-center gap-2 text-[13px]">
            <Switch checked={tiny} onCheckedChange={setTiny} disabled={running} aria-label="Build it tiny" />
            Build it tiny: load a small random-weight copy of its architecture the way bolo would
          </label>
          {report && (
            <div role="status" data-testid="check-report">
              <p className={cn('mb-2 text-[13px] font-medium', report.ok ? 'text-good-text' : 'text-critical-text')}>
                {report.ok ? 'No problems found.' : 'Something would fail:'}
              </p>
              <ul className="grid gap-1.5 text-xs leading-5">
                {report.steps.map((s, i) => (
                  <li key={`${s.name}-${i}`} className="grid grid-cols-[3.5rem_7rem_minmax(0,1fr)] gap-2">
                    <span className={cn('font-medium uppercase', STATUS_CLASS[s.status] ?? '')}>{s.status}</span>
                    <span className="text-muted-foreground">{s.name}</span>
                    <span className="break-words">{s.detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {error && (
            <p role="alert" className="text-[13px] text-critical-text">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)} disabled={running}>
              Close
            </Button>
            <Button onClick={() => void run()} disabled={running} data-testid="check-run">
              <ClipboardCheck /> {running ? 'Checking…' : report ? 'Check again' : 'Run the check'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
