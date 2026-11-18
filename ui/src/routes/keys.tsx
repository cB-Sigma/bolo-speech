import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Ban, Check, Copy, KeyRound, Pencil, Plus, ShieldAlert, Trash2 } from 'lucide-react'
import { useId, useState } from 'react'
import { toast } from 'sonner'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
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
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { getApiKey, isDemo, setApiKey } from '@/lib/api/config'
import type { ApiKeyInfo } from '@/lib/api/types'
import { fmtInt } from '@/lib/format'
import {
  DOCS_KEYS,
  KEY_SCOPES,
  keyBody,
  keyForm,
  keyState,
  limitsText,
  PERIOD_LABEL,
  quotaUsed,
  SCOPE_LABEL,
  type KeyForm,
} from '@/lib/keys'
import { cn } from '@/lib/utils'

function KeyDialog({
  existing,
  scopeHelp,
  firstKey,
  onClose,
  onCreated,
}: {
  existing: ApiKeyInfo | null
  scopeHelp: Record<string, string>
  firstKey: boolean
  onClose: () => void
  onCreated: (key: ApiKeyInfo) => void
}) {
  const qc = useQueryClient()
  const id = useId()
  const [form, setForm] = useState<KeyForm>(() =>
    firstKey && !existing ? { ...keyForm(), name: 'console', scopes: ['admin'] } : keyForm(existing),
  )
  const [saving, setSaving] = useState(false)
  const set = <K extends keyof KeyForm>(k: K, v: KeyForm[K]) => setForm((f) => ({ ...f, [k]: v }))
  const { body, problem } = keyBody(form)
  const needsAdmin = firstKey && !existing && !form.scopes.includes('admin')

  const save = async () => {
    if (problem || needsAdmin) return
    setSaving(true)
    try {
      if (existing) {
        const { expires_in_days: _ignored, ...changes } = body
        await api.updateKey(existing.id, changes)
        toast.success(`${body.name} saved`)
      } else {
        onCreated(await api.createKey(body))
      }
      await qc.invalidateQueries({ queryKey: ['keys'] })
      await qc.invalidateQueries({ queryKey: ['info'] })
      onClose()
    } catch (err) {
      toast.error('Could not save the key', { description: describeError(err) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{existing ? `Edit ${existing.name}` : 'Create an API key'}</DialogTitle>
          <DialogDescription>
            {existing
              ? 'Changes apply to the next request made with this key.'
              : 'The key is shown once, when it is created. The server keeps only a hash of it.'}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-name`}>Name</Label>
            <Input
              id={`${id}-name`}
              value={form.name}
              maxLength={80}
              placeholder="mobile app"
              onChange={(e) => set('name', e.target.value)}
            />
          </div>
          <fieldset className="grid gap-2">
            <legend className="mb-1 text-sm font-medium">Scopes</legend>
            {KEY_SCOPES.map((scope) => (
              <label key={scope} className="flex items-start gap-2 text-sm" htmlFor={`${id}-scope-${scope}`}>
                <Checkbox
                  id={`${id}-scope-${scope}`}
                  className="mt-0.5"
                  checked={form.scopes.includes(scope)}
                  onCheckedChange={(on) =>
                    set('scopes', on ? [...form.scopes, scope] : form.scopes.filter((s) => s !== scope))
                  }
                />
                <span>
                  {SCOPE_LABEL[scope]}
                  <span className="block text-xs text-muted-foreground">{scopeHelp[scope]}</span>
                </span>
              </label>
            ))}
          </fieldset>
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-rate`}>Requests per minute</Label>
              <Input
                id={`${id}-rate`}
                inputMode="decimal"
                placeholder="no limit"
                value={form.rateLimit}
                onChange={(e) => set('rateLimit', e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-burst`}>Burst</Label>
              <Input
                id={`${id}-burst`}
                inputMode="numeric"
                placeholder="= per minute"
                value={form.burst}
                onChange={(e) => set('burst', e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-quota`}>Quota (requests)</Label>
              <Input
                id={`${id}-quota`}
                inputMode="numeric"
                placeholder="no quota"
                value={form.quota}
                onChange={(e) => set('quota', e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-period`}>Quota period</Label>
              <Select value={form.quotaPeriod} onValueChange={(v) => set('quotaPeriod', v)}>
                <SelectTrigger id={`${id}-period`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(PERIOD_LABEL).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label} {value !== 'total' && '(UTC)'}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {!existing && (
              <div className="grid gap-1.5">
                <Label htmlFor={`${id}-expires`}>Expires after (days)</Label>
                <Input
                  id={`${id}-expires`}
                  inputMode="decimal"
                  placeholder="never"
                  value={form.expiresInDays}
                  onChange={(e) => set('expiresInDays', e.target.value)}
                />
              </div>
            )}
          </div>
          {(problem || needsAdmin) && (
            <p className="text-xs text-critical-text" role="alert">
              {problem ?? 'The first key turns authentication on, so it needs the Admin scope.'}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={() => void save()} disabled={saving || Boolean(problem) || needsAdmin}>
            {existing ? 'Save' : 'Create key'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function NewKey({ created, onClose }: { created: ApiKeyInfo; onClose: () => void }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(created.key ?? '')
      setCopied(true)
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Key created: {created.name}</DialogTitle>
          <DialogDescription>Copy it now. It cannot be shown again.</DialogDescription>
        </DialogHeader>
        <div className="flex items-stretch overflow-hidden rounded-lg border border-border">
          <code className="min-w-0 flex-1 overflow-x-auto px-3 py-2 num text-xs" data-testid="new-key">
            {created.key}
          </code>
          <Button variant="ghost" className="rounded-none border-l" onClick={() => void copy()} aria-label="Copy">
            {copied ? <Check /> : <Copy />}
          </Button>
        </div>
        <DialogFooter>
          <Button onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function KeyRow({
  k,
  you,
  onEdit,
  onRevoke,
  onDelete,
}: {
  k: ApiKeyInfo
  you: boolean
  onEdit: () => void
  onRevoke: () => void
  onDelete: () => void
}) {
  const state = keyState(k)
  const used = quotaUsed(k)
  return (
    <li className="grid gap-2 px-4 py-3" data-testid={`key-${k.id}`}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-medium">{k.name}</span>
        <code className="num text-xs text-muted-foreground">{k.hint}</code>
        {you && <span className="text-xs text-muted-foreground">(this console)</span>}
        <span
          className={cn(
            'text-xs',
            state === 'active' ? 'text-good-text' : state === 'revoked' ? 'text-critical-text' : 'text-warning-text',
          )}
        >
          {state}
        </span>
        <div className="ml-auto flex gap-1">
          {state === 'active' && (
            <>
              <Button size="sm" variant="outline" onClick={onEdit}>
                <Pencil /> Edit
              </Button>
              <Button size="sm" variant="ghost" onClick={onRevoke}>
                <Ban /> Revoke
              </Button>
            </>
          )}
          <Button size="icon-sm" variant="ghost" aria-label={`Delete ${k.name}`} onClick={onDelete}>
            <Trash2 />
          </Button>
        </div>
      </div>
      <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
        <div className="flex gap-1.5">
          <dt>Scopes</dt>
          <dd className="text-foreground">{k.scopes.map((s) => SCOPE_LABEL[s] ?? s).join(', ')}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Limits</dt>
          <dd className="text-foreground">{limitsText(k)}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Requests</dt>
          <dd className="num text-foreground">{fmtInt(k.usage.requests)}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Refused</dt>
          <dd className={cn('num', k.usage.rejected ? 'text-warning-text' : 'text-foreground')}>
            {fmtInt(k.usage.rejected)}
          </dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Last used</dt>
          <dd className="text-foreground">{k.usage.last_used_at?.replace('T', ' ').replace('Z', ' UTC') ?? 'never'}</dd>
        </div>
        {k.expires_at && (
          <div className="flex gap-1.5">
            <dt>Expires</dt>
            <dd className="text-foreground">{k.expires_at.slice(0, 10)}</dd>
          </div>
        )}
      </dl>
      {used != null && (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <Progress value={used * 100} className="h-1.5 max-w-60" aria-label="Quota used" />
          <span className="num">
            {fmtInt(k.usage.period_requests)} of {fmtInt(k.quota)} {PERIOD_LABEL[k.quota_period]}
          </span>
        </div>
      )}
    </li>
  )
}

/** API keys: one per client, each with scopes, a rate limit and a quota, and its usage. */
export default function KeysPage() {
  const qc = useQueryClient()
  const q = useQuery({
    queryKey: ['keys'],
    queryFn: ({ signal }) => api.keys(signal),
    refetchInterval: 15_000,
    retry: false,
    enabled: !isDemo,
  })
  const [editing, setEditing] = useState<ApiKeyInfo | 'new' | null>(null)
  const [created, setCreated] = useState<ApiKeyInfo | null>(null)
  const [confirm, setConfirm] = useState<{ key: ApiKeyInfo; remove: boolean } | null>(null)
  const list = q.data?.data ?? []
  const firstKey = q.data ? !q.data.auth : false

  const revoke = async (k: ApiKeyInfo, remove: boolean) => {
    try {
      await api.revokeKey(k.id, remove)
      await qc.invalidateQueries({ queryKey: ['keys'] })
      toast.success(`${k.name} ${remove ? 'deleted' : 'revoked'}`)
    } catch (err) {
      toast.error(`Could not ${remove ? 'delete' : 'revoke'} ${k.name}`, { description: describeError(err) })
    }
  }

  if (isDemo) {
    return (
      <Page title="API keys">
        <NeedsServer title="API keys are kept by a bolo server">
          A server gives each client a key of its own, with scopes, a rate limit and a quota. This demo runs in your
          browser only.
        </NeedsServer>
      </Page>
    )
  }

  return (
    <Page title="API keys">
      {q.data && !q.data.auth && (
        <Alert className="mb-4">
          <ShieldAlert />
          <AlertTitle>Authentication is off</AlertTitle>
          <AlertDescription>
            Anyone who can reach this server can use it. The first key you create turns authentication on; it needs the
            Admin scope and this console keeps it, so you stay signed in.{' '}
            <a href={DOCS_KEYS} target="_blank" rel="noreferrer" className="underline underline-offset-2">
              How keys work
            </a>
          </AlertDescription>
        </Alert>
      )}
      {q.data?.root_key && (
        <p className="mb-4 text-sm text-muted-foreground">
          The server&apos;s <code>api_key</code> setting is a key with every scope; it is not listed here.
        </p>
      )}
      <Panel
        title="Keys"
        description={q.data ? `${list.filter((k) => k.active).length} active of ${list.length}` : undefined}
        bodyClassName="p-0"
        actions={
          <Button size="sm" onClick={() => setEditing('new')} disabled={!q.data}>
            <Plus /> Create key
          </Button>
        }
      >
        {q.isLoading ? (
          <Skeleton className="m-4 h-24" />
        ) : q.isError ? (
          <ErrorState
            title="Could not load the keys"
            error={`${describeError(q.error)} Managing keys needs a key with the Admin scope.`}
            onRetry={() => void q.refetch()}
          />
        ) : list.length === 0 ? (
          <Empty icon={KeyRound} title="No keys yet">
            Create a key for each app or person that calls this server, with only the scopes it needs.
          </Empty>
        ) : (
          <ul className="divide-y divide-border" data-testid="keys">
            {list.map((k) => (
              <KeyRow
                key={k.id}
                k={k}
                you={q.data?.you === k.id}
                onEdit={() => setEditing(k)}
                onRevoke={() => setConfirm({ key: k, remove: false })}
                onDelete={() => setConfirm({ key: k, remove: true })}
              />
            ))}
          </ul>
        )}
      </Panel>

      {editing && (
        <KeyDialog
          key={editing === 'new' ? 'new' : editing.id}
          existing={editing === 'new' ? null : editing}
          scopeHelp={q.data?.scopes ?? {}}
          firstKey={firstKey}
          onClose={() => setEditing(null)}
          onCreated={(k) => {
            if (firstKey || !getApiKey()) setApiKey(k.key ?? null)
            setCreated(k)
          }}
        />
      )}
      {created && <NewKey created={created} onClose={() => setCreated(null)} />}

      <Dialog open={confirm != null} onOpenChange={(o) => !o && setConfirm(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {confirm?.remove ? 'Delete' : 'Revoke'} {confirm?.key.name}?
            </DialogTitle>
            <DialogDescription>
              {confirm?.remove
                ? 'The key and its usage counters are removed. Requests with it fail from now on.'
                : 'Requests with this key fail from now on. It stays listed with its usage.'}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (confirm) void revoke(confirm.key, confirm.remove)
                setConfirm(null)
              }}
            >
              {confirm?.remove ? <Trash2 /> : <Ban />} {confirm?.remove ? 'Delete' : 'Revoke'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Page>
  )
}
