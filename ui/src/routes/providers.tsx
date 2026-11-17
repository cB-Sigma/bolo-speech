import { useQueryClient } from '@tanstack/react-query'
import { CircleCheck, CircleX, Cloud, ExternalLink, KeyRound, Loader2, Pencil, Plus, Trash2, X } from 'lucide-react'
import { useId, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError, useCatalog, useProviders } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { Provider, ProviderKind, ProviderTest } from '@/lib/api/types'
import { CLOUD_NOTICE } from '@/lib/capabilities'
import { fmtCost, PRICE_UNIT_LABEL, PRICE_UNITS, providerTotals, type PriceRow, type PriceUnit } from '@/lib/cost'
import { fmtInt, fmtMs } from '@/lib/format'
import {
  catalogName,
  DOCS_PROVIDERS,
  PROVIDER_KIND_LABEL,
  PROVIDER_KIND_NOTE,
  PROVIDER_NAME_RE,
  providerForm,
  providerUpdate,
  type ProviderForm,
} from '@/lib/providers'
import { cn } from '@/lib/utils'

function StateBadge({ p }: { p: Provider }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 text-xs',
        p.configured ? 'text-good-text' : 'text-muted-foreground',
      )}
      data-testid={`provider-state-${p.name}`}
    >
      {p.configured ? <CircleCheck className="size-3.5" aria-hidden /> : <KeyRound className="size-3.5" aria-hidden />}
      {p.configured ? (p.api_key_set ? 'Configured' : 'Configured (no key needed)') : 'No API key'}
    </span>
  )
}

function TestResult({ result }: { result: ProviderTest }) {
  const ok = result.ok
  return (
    <p
      className={cn('flex items-start gap-1.5 text-xs leading-5', ok ? 'text-good-text' : 'text-critical-text')}
      role="status"
      data-testid={`provider-test-${result.name}`}
    >
      {ok ? (
        <CircleCheck className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      ) : (
        <CircleX className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      )}
      <span>
        {ok
          ? result.checked === 'credentials'
            ? 'The key was accepted'
            : 'The address answers (the key itself could not be checked)'
          : 'Failed'}
        {' · '}
        <span className="num">{fmtMs(result.latency_ms)}</span>
        {' · '}
        {result.message}
      </span>
    </p>
  )
}

function ProviderDialog({
  open,
  existing,
  kinds,
  taken,
  onClose,
}: {
  open: boolean
  existing: Provider | null
  kinds: ProviderKind[]
  taken: string[]
  onClose: () => void
}) {
  const qc = useQueryClient()
  const id = useId()
  const [form, setForm] = useState<ProviderForm>(() => providerForm(existing, kinds))
  const [saving, setSaving] = useState(false)
  const set = <K extends keyof ProviderForm>(key: K, value: ProviderForm[K]) => setForm((f) => ({ ...f, [key]: value }))
  const kind = kinds.find((k) => k.kind === form.kind)
  const nameProblem = existing
    ? null
    : form.name && !PROVIDER_NAME_RE.test(form.name)
      ? 'Lower-case letters, digits, - and _ only.'
      : taken.includes(form.name)
        ? 'A provider with this name exists.'
        : null
  const { body, problem } = providerUpdate(form)

  const save = async () => {
    if (problem || nameProblem || !form.name) return
    setSaving(true)
    try {
      await api.putProvider(form.name, body)
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['providers'] }),
        qc.invalidateQueries({ queryKey: ['catalog'] }),
        qc.invalidateQueries({ queryKey: ['info'] }),
      ])
      toast.success(existing ? `${form.name} saved` : `${form.name} added`, {
        description: 'Its cloud models are now listed as installed on the Models screen.',
      })
      onClose()
    } catch (err) {
      toast.error('Could not save the provider', { description: describeError(err) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{existing ? `Edit ${existing.name}` : 'Add a provider'}</DialogTitle>
          <DialogDescription>
            {CLOUD_NOTICE}. A request reaches a provider only when it names one of its models, or when a model&apos;s
            fallback list does.
          </DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            void save()
          }}
        >
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-kind`} className="text-xs">
              API
            </Label>
            <Select value={form.kind} onValueChange={(v) => set('kind', v)}>
              <SelectTrigger id={`${id}-kind`} className="h-8" aria-label="API">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                {kinds.map((k) => (
                  <SelectItem key={k.kind} value={k.kind}>
                    {PROVIDER_KIND_LABEL[k.kind] ?? k.kind}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{PROVIDER_KIND_NOTE[form.kind]}</p>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-name`} className="text-xs">
              Name
            </Label>
            <Input
              id={`${id}-name`}
              value={form.name}
              onChange={(e) => set('name', e.target.value.toLowerCase())}
              placeholder={catalogName(form.kind)}
              className="h-8 num"
              disabled={existing != null}
              aria-invalid={nameProblem ? true : undefined}
              autoComplete="off"
            />
            <p className={cn('text-xs', nameProblem ? 'text-critical-text' : 'text-muted-foreground')}>
              {nameProblem ??
                `The catalog's ${PROVIDER_KIND_LABEL[form.kind] ?? form.kind} models look for the name ${catalogName(form.kind)}; any other model is used as name:model.`}
            </p>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-url`} className="text-xs">
              Address
            </Label>
            <Input
              id={`${id}-url`}
              value={form.baseUrl}
              onChange={(e) => set('baseUrl', e.target.value)}
              placeholder={kind?.default_base_url}
              className="h-8 num"
              inputMode="url"
              autoComplete="off"
            />
            <p className="text-xs text-muted-foreground">Leave empty for the provider&apos;s own address.</p>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-key`} className="text-xs">
              API key
            </Label>
            <Input
              id={`${id}-key`}
              type="password"
              value={form.apiKey}
              onChange={(e) => setForm((f) => ({ ...f, apiKey: e.target.value, clearKey: false }))}
              placeholder={existing?.api_key_set ? 'A key is stored. Type a new one to replace it.' : 'Paste the key'}
              className="h-8 num"
              autoComplete="new-password"
              data-testid="provider-key"
            />
            <p className="text-xs text-muted-foreground">
              Write-only: the server stores it and never shows it again, here or through the API.
              {existing?.api_key_set && !form.apiKey && (
                <>
                  {' '}
                  <button
                    type="button"
                    className="rounded-sm underline underline-offset-2 hover:text-foreground"
                    onClick={() => set('clearKey', !form.clearKey)}
                    aria-pressed={form.clearKey}
                  >
                    {form.clearKey ? 'Keep the stored key' : 'Remove the stored key'}
                  </button>
                  {form.clearKey && <span className="text-warning-text"> · the key will be removed on save</span>}
                </>
              )}
            </p>
          </div>
          <fieldset className="grid gap-2">
            <legend className="mb-1 text-xs font-medium">
              Prices <span className="font-normal text-muted-foreground">(optional, for cost estimates)</span>
            </legend>
            {form.prices.map((row, i) => {
              const edit = (patch: Partial<PriceRow>) =>
                set(
                  'prices',
                  form.prices.map((r, j) => (j === i ? { ...r, ...patch } : r)),
                )
              return (
                <div key={i} className="grid grid-cols-[minmax(0,5fr)_minmax(0,6fr)_4.5rem_3.5rem_auto] gap-1.5">
                  <Input
                    value={row.model}
                    onChange={(e) => edit({ model: e.target.value })}
                    placeholder="model, or *"
                    aria-label={`Price ${i + 1}: model`}
                    className="h-8 num"
                  />
                  <Select value={row.unit} onValueChange={(v) => edit({ unit: v as PriceUnit })}>
                    <SelectTrigger className="h-8" aria-label={`Price ${i + 1}: unit`}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent position="popper" align="start">
                      {PRICE_UNITS.map((u) => (
                        <SelectItem key={u} value={u}>
                          {PRICE_UNIT_LABEL[u]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    value={row.amount}
                    onChange={(e) => edit({ amount: e.target.value })}
                    placeholder="0.006"
                    aria-label={`Price ${i + 1}: amount`}
                    className="h-8 num"
                    inputMode="decimal"
                  />
                  <Input
                    value={row.currency}
                    onChange={(e) => edit({ currency: e.target.value.toUpperCase().slice(0, 3) })}
                    aria-label={`Price ${i + 1}: currency`}
                    className="h-8 num"
                  />
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    aria-label={`Remove price ${i + 1}`}
                    onClick={() =>
                      set(
                        'prices',
                        form.prices.filter((_, j) => j !== i),
                      )
                    }
                  >
                    <X />
                  </Button>
                </div>
              )
            })}
            <div>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() =>
                  set('prices', [...form.prices, { model: '', unit: 'per_minute', amount: '', currency: 'USD' }])
                }
              >
                <Plus /> Add a price
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              What you pay, by the provider&apos;s model name. bolo ships no prices; the cost it shows is an estimate
              from this table, never a bill.
            </p>
          </fieldset>
          {problem && (
            <p className="text-xs text-critical-text" role="alert">
              {problem}
            </p>
          )}
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving || !form.name || nameProblem != null || problem != null}>
              {saving && <Loader2 className="animate-spin" />}
              {existing ? 'Save' : 'Add provider'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function ProviderRow({
  p,
  models,
  canEdit,
  onEdit,
  onDelete,
}: {
  p: Provider
  models: number
  canEdit: boolean
  onEdit: () => void
  onDelete: () => void
}) {
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<ProviderTest | null>(null)
  const totals = providerTotals(p.stats)
  const spend = Object.entries(totals.cost)
  const test = async () => {
    setTesting(true)
    setResult(null)
    try {
      setResult(await api.testProvider(p.name))
    } catch (err) {
      setResult({ name: p.name, ok: false, message: describeError(err), code: null, checked: null, latency_ms: 0 })
    } finally {
      setTesting(false)
    }
  }
  return (
    <li className="grid gap-2 px-4 py-3" data-testid={`provider-${p.name}`}>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="num text-sm font-medium">{p.name}</span>
            <span className="text-xs text-muted-foreground">{PROVIDER_KIND_LABEL[p.kind] ?? p.kind}</span>
            <StateBadge p={p} />
            {!p.editable && (
              <span className="text-xs text-muted-foreground" title="Defined in the server's configuration">
                from the configuration
              </span>
            )}
          </div>
          <div className="mt-0.5 truncate num text-xs text-muted-foreground">{p.base_url}</div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button size="sm" variant="outline" onClick={() => void test()} disabled={testing}>
            {testing ? <Loader2 className="animate-spin" /> : <CircleCheck />} Test
          </Button>
          {p.editable && canEdit && (
            <>
              <Button size="sm" variant="outline" onClick={onEdit}>
                <Pencil /> Edit
              </Button>
              <Button size="icon-sm" variant="ghost" aria-label={`Remove ${p.name}`} onClick={onDelete}>
                <Trash2 />
              </Button>
            </>
          )}
        </div>
      </div>
      <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
        <div className="flex gap-1.5">
          <dt>Catalog models</dt>
          <dd className="num text-foreground">{models}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Requests</dt>
          <dd className="num text-foreground">{fmtInt(totals.requests)}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Errors</dt>
          <dd className={cn('num', totals.errors ? 'text-critical-text' : 'text-foreground')}>
            {fmtInt(totals.errors)}
          </dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Estimated spend</dt>
          <dd className="num text-foreground">
            {spend.length
              ? spend.map(([currency, amount]) => fmtCost({ amount, currency })).join(' + ')
              : Object.keys(p.prices).length
                ? '–'
                : 'no price table'}
          </dd>
        </div>
      </dl>
      {p.terms_url && (
        <a
          href={p.terms_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex w-fit items-center gap-1 rounded-sm text-xs text-muted-foreground hover:text-foreground"
        >
          <ExternalLink className="size-3" aria-hidden /> Terms
        </a>
      )}
      {result && <TestResult result={result} />}
    </li>
  )
}

/** Cloud providers: the accounts behind `engine: remote` models. Add, edit, test and remove them. */
export default function ProvidersPage() {
  const q = useProviders()
  const catalog = useCatalog()
  const qc = useQueryClient()
  const [editing, setEditing] = useState<Provider | 'new' | null>(null)
  const [toDelete, setToDelete] = useState<Provider | null>(null)
  const list = q.data?.providers ?? []
  const kinds = q.data?.kinds ?? []
  const canEdit = q.data?.editable ?? false
  const modelsOf = useMemo(() => {
    const count = new Map<string, number>()
    for (const m of catalog.data?.models ?? []) {
      if (m.provider?.name) count.set(m.provider.name, (count.get(m.provider.name) ?? 0) + 1)
    }
    return count
  }, [catalog.data])

  const remove = async (p: Provider) => {
    try {
      await api.deleteProvider(p.name)
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['providers'] }),
        qc.invalidateQueries({ queryKey: ['catalog'] }),
        qc.invalidateQueries({ queryKey: ['info'] }),
      ])
      toast.success(`${p.name} removed`)
    } catch (err) {
      toast.error(`Could not remove ${p.name}`, { description: describeError(err) })
    }
  }

  if (isDemo) {
    return (
      <Page title="Providers">
        <NeedsServer title="Cloud providers are set up on a bolo server">
          A server can hand requests to cloud speech APIs (OpenAI, Sarvam, ElevenLabs, Deepgram, Soniox) next to the
          models it runs itself. This demo runs in your browser only and talks to none of them.
        </NeedsServer>
      </Page>
    )
  }

  return (
    <Page title="Providers">
      <Alert className="mb-4" data-testid="cloud-notice">
        <Cloud />
        <AlertTitle>{q.data?.notice ? sentenceCase(q.data.notice) : CLOUD_NOTICE}</AlertTitle>
        <AlertDescription>
          <span>
            A provider is an account at a cloud speech API. Its models appear on the Models screen marked
            &quot;Cloud&quot;, and are used only when a request names one, a default points at one, or a model&apos;s
            fallback list does. Keys are stored on the server and never shown again.{' '}
            <a href={DOCS_PROVIDERS} target="_blank" rel="noreferrer" className="underline underline-offset-2">
              How providers work
            </a>
          </span>
        </AlertDescription>
      </Alert>
      {q.data && !canEdit && (
        <Alert className="mb-4">
          <KeyRound />
          <AlertTitle>Providers cannot be changed from here</AlertTitle>
          <AlertDescription>
            {q.data.edit_blocked_reason ?? 'The server does not allow it.'} Set them in the server&apos;s configuration
            instead.
          </AlertDescription>
        </Alert>
      )}
      <Panel
        title="Providers"
        description={q.data ? `${list.filter((p) => p.configured).length} of ${list.length} configured` : undefined}
        bodyClassName="p-0"
        actions={
          <Button size="sm" onClick={() => setEditing('new')} disabled={!canEdit || !q.data}>
            <Plus /> Add provider
          </Button>
        }
      >
        {!q.offered ? (
          <Empty icon={Cloud} title="This server has no cloud providers">
            It is older than the remote engine. Upgrade bolo to connect cloud speech APIs.
          </Empty>
        ) : q.isLoading ? (
          <Skeleton className="m-4 h-24" />
        ) : q.isError ? (
          <ErrorState
            title="Could not load the providers"
            error={describeError(q.error)}
            onRetry={() => void q.refetch()}
          />
        ) : list.length === 0 ? (
          <Empty icon={Cloud} title="No providers yet">
            Nothing leaves this server. Add a provider to use its cloud models next to the local ones.
          </Empty>
        ) : (
          <ul className="divide-y divide-border" data-testid="providers">
            {list.map((p) => (
              <ProviderRow
                key={p.name}
                p={p}
                models={modelsOf.get(p.name) ?? 0}
                canEdit={canEdit}
                onEdit={() => setEditing(p)}
                onDelete={() => setToDelete(p)}
              />
            ))}
          </ul>
        )}
      </Panel>

      {editing && (
        <ProviderDialog
          key={editing === 'new' ? 'new' : editing.name}
          open
          existing={editing === 'new' ? null : editing}
          kinds={kinds}
          taken={list.map((p) => p.name)}
          onClose={() => setEditing(null)}
        />
      )}

      <Dialog open={toDelete != null} onOpenChange={(o) => !o && setToDelete(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Remove <span className="num">{toDelete?.name}</span>?
            </DialogTitle>
            <DialogDescription>
              The server forgets the provider and its key. Its cloud models stay in the catalog but can no longer be
              used until a provider is added again.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (toDelete) void remove(toDelete)
                setToDelete(null)
              }}
            >
              <Trash2 /> Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Page>
  )
}

function sentenceCase(text: string): string {
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}`
}
