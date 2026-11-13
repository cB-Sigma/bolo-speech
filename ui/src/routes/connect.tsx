import { BookOpen, Check, CircleCheck, CircleX, Copy, ExternalLink, KeyRound, TriangleAlert } from 'lucide-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { askForApiKey } from '@/components/layout/connection'
import { DOCS_URL, Page } from '@/components/layout/app-shell'
import { ModelPicker } from '@/components/model-picker'
import { ErrorState, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { describeError, useInfo, useSelectedModel } from '@/hooks/use-api'
import { useStore } from '@/hooks/use-store'
import { useSupport } from '@/hooks/use-support'
import { ComponentsCard } from '@/components/component-install'
import { NeedsServer } from '@/components/demo'
import { getApiKey, isDemo, serverBase } from '@/lib/api/config'
import { prefs } from '@/lib/prefs'
import { buildSnippets } from '@/lib/snippets'
import { cn } from '@/lib/utils'

function CodeBlock({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }
  return (
    <figure className="overflow-hidden rounded-lg border border-border">
      <figcaption className="flex h-9 items-center justify-between gap-2 border-b border-border bg-muted/40 pr-1 pl-3 text-xs text-muted-foreground">
        <span className="truncate">{title}</span>
        <Button size="xs" variant="ghost" onClick={() => void copy()} aria-label={`Copy: ${title}`}>
          {copied ? <Check /> : <Copy />}
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </figcaption>
      <pre className="overflow-x-auto bg-background/60 p-3 num text-[12.5px] leading-5 text-foreground" tabIndex={0}>
        <code>{code}</code>
      </pre>
    </figure>
  )
}

function Check3({ ok, label, children }: { ok: boolean | null; label: string; children: React.ReactNode }) {
  const Icon = ok == null ? TriangleAlert : ok ? CircleCheck : CircleX
  return (
    <li className="flex items-start gap-2.5 py-2">
      <Icon
        className={cn(
          'mt-0.5 size-4 shrink-0',
          ok == null ? 'text-muted-foreground' : ok ? 'text-good-text' : 'text-muted-foreground',
        )}
        aria-hidden
      />
      <div className="min-w-0 flex-1 text-[13px] leading-5">
        <div className="flex items-baseline justify-between gap-3">
          <span className="font-medium">{label}</span>
          <span className="shrink-0 text-xs text-muted-foreground">
            {ok == null ? 'Checking' : ok ? 'Available' : 'Unavailable'}
          </span>
        </div>
        <p className="text-xs leading-4 text-muted-foreground">{children}</p>
      </div>
    </li>
  )
}

export default function ConnectPage() {
  const info = useInfo()
  const stt = useSelectedModel('stt')
  const tts = useSelectedModel('tts')
  const p = useStore(prefs)
  const support = useSupport()
  // The demo has no server of its own; show snippets for the default local address.
  const baseUrl = isDemo ? 'http://127.0.0.1:8000' : serverBase().replace(/\/$/, '')
  const voice = tts.model?.voices.find((v) => v.id === p.ttsVoice)?.id ?? tts.model?.voices[0]?.id ?? 'default'
  const snippets = useMemo(
    () =>
      buildSnippets({
        baseUrl,
        sttModel: stt.model?.id ?? 'moonshine-tiny-en',
        ttsModel: tts.model?.id ?? 'kokoro-82m',
        voice,
        auth: Boolean(info.data?.auth_required),
      }),
    [baseUrl, stt.model, tts.model, voice, info.data],
  )

  return (
    <Page title="Connect">
      {isDemo && (
        <NeedsServer title="These snippets talk to a bolo server" className="mb-4">
          The live demo has no API behind it. Install bolo, start it, and the snippets below work as written against{' '}
          <span className="num text-foreground">http://127.0.0.1:8000</span>, including the Wyoming port for Home
          Assistant.
        </NeedsServer>
      )}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <Panel
          title="Client snippets"
          description={
            <span>
              Pre-filled for <span className="num">{baseUrl}</span>
            </span>
          }
          bodyClassName="grid gap-4"
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <Label htmlFor="snip-stt" className="text-xs">
                Speech-to-text model
              </Label>
              <ModelPicker
                id="snip-stt"
                models={stt.models}
                value={stt.model?.id ?? null}
                onChange={stt.select}
                loading={stt.isLoading}
                className="w-full"
                label="Speech-to-text model"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="snip-tts" className="text-xs">
                Text-to-speech model
              </Label>
              <ModelPicker
                id="snip-tts"
                models={tts.models}
                value={tts.model?.id ?? null}
                onChange={tts.select}
                loading={tts.isLoading}
                className="w-full"
                label="Text-to-speech model"
              />
            </div>
          </div>
          <Tabs defaultValue="curl" className="min-w-0">
            <TabsList className="max-w-full justify-start overflow-x-auto">
              {snippets.map((s) => (
                <TabsTrigger key={s.id} value={s.id} className="text-xs">
                  {s.label}
                </TabsTrigger>
              ))}
            </TabsList>
            {snippets.map((s) => (
              <TabsContent key={s.id} value={s.id} className="mt-3 grid gap-3">
                {s.note && <p className="text-[13px] leading-5 text-muted-foreground">{s.note}</p>}
                {s.blocks.map((b) => (
                  <CodeBlock key={b.title} title={b.title} code={b.code} />
                ))}
              </TabsContent>
            ))}
          </Tabs>
          <p className="text-xs leading-5 text-muted-foreground">
            The OpenAI-compatible routes live under <code className="num text-foreground">/v1</code>; model management,
            streaming and metrics live under <code className="num text-foreground">/api/v1</code>. Prometheus metrics
            are at <code className="num text-foreground">/metrics</code>.
          </p>
        </Panel>

        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title={isDemo ? 'This demo' : 'Server'}
            actions={
              <Button size="sm" variant="ghost" asChild>
                <a href={DOCS_URL} target="_blank" rel="noreferrer">
                  <BookOpen /> Docs <ExternalLink className="size-3 text-muted-foreground" />
                </a>
              </Button>
            }
          >
            {info.isLoading ? (
              <div className="grid gap-2">
                {Array.from({ length: 6 }, (_, i) => (
                  <Skeleton key={i} className="h-4" />
                ))}
              </div>
            ) : info.isError || !info.data ? (
              <ErrorState
                title="Server details unavailable"
                error={describeError(info.error)}
                onRetry={() => void info.refetch()}
                className="py-4"
              />
            ) : (
              <>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
                  <dt className="text-muted-foreground">Version</dt>
                  <dd className="text-right num">
                    {info.data.name} {info.data.version}
                  </dd>
                  <dt className="text-muted-foreground">Address</dt>
                  <dd className="truncate text-right num" title={baseUrl}>
                    {baseUrl.replace(/^https?:\/\//, '')}
                  </dd>
                  <dt className="text-muted-foreground">Device</dt>
                  <dd className="text-right num">
                    {info.data.device} · {info.data.cpu_count} cores
                  </dd>
                  <dt className="text-muted-foreground">Platform</dt>
                  <dd className="text-right num">{info.data.platform}</dd>
                  <dt className="text-muted-foreground">Data directory</dt>
                  <dd className="truncate text-right num" title={info.data.data_dir}>
                    {info.data.data_dir}
                  </dd>
                  <dt className="text-muted-foreground">Authentication</dt>
                  <dd className="flex items-center justify-end gap-2 text-right">
                    {info.data.auth_required ? (getApiKey() ? 'API key set' : 'API key required') : 'Off'}
                    {info.data.auth_required && (
                      <Button size="xs" variant="outline" onClick={askForApiKey}>
                        <KeyRound /> {getApiKey() ? 'Change' : 'Enter'}
                      </Button>
                    )}
                  </dd>
                </dl>
                <h3 className="mt-4 mb-0.5 eyebrow">Optional components</h3>
                <ComponentsCard bare />
                <h3 className="mt-3 mb-1.5 eyebrow">Engines</h3>
                <ul className="divide-y divide-border border-t border-border">
                  {info.data.engines.map((e) => (
                    <li key={e.id} className="py-2 text-[13px]">
                      <div className="flex items-center justify-between gap-3">
                        <span className="flex items-center gap-2">
                          <span
                            className={cn('size-1.5 rounded-full', e.available ? 'bg-good' : 'bg-chart-mute')}
                            aria-hidden
                          />
                          <span className="num">{e.id}</span>
                        </span>
                        <span className="num text-xs text-muted-foreground">
                          {e.available ? (e.version ?? 'available') : 'unavailable'}
                        </span>
                      </div>
                      {!e.available && e.reason && (
                        <p className="mt-0.5 pl-3.5 text-xs leading-4 text-muted-foreground">{e.reason}</p>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </Panel>

          <Panel title="This browser" description="What the playground can use here" bodyClassName="py-2">
            <ul className="divide-y divide-border" data-testid="support-list">
              <Check3 ok={support.getUserMedia && support.secureContext} label="Microphone">
                {!support.secureContext
                  ? 'Needs HTTPS or localhost. Use file upload or samples instead.'
                  : support.micPermission === 'denied'
                    ? 'Access is blocked for this site. Allow it in the address bar.'
                    : support.micPermission === 'granted'
                      ? 'Permission granted.'
                      : 'You will be asked for permission on first use.'}
              </Check3>
              <Check3 ok={support.audioWorklet} label="AudioWorklet">
                Low-latency capture off the main thread.
              </Check3>
              <Check3 ok={support.wasm && support.wasmSimd} label="WebAssembly SIMD">
                {support.wasm
                  ? 'Required for the in-browser model to run at a usable speed.'
                  : 'Not supported: in-browser mode is disabled.'}
              </Check3>
              <Check3 ok={support.wasmThreads} label="WebAssembly threads">
                {support.wasmThreads
                  ? `Cross-origin isolated; up to ${Math.min(4, support.hardwareConcurrency)} threads.`
                  : 'Off because the page is not cross-origin isolated (COOP and COEP headers). The in-browser model runs on one thread.'}
              </Check3>
              <Check3 ok={support.webgpuAdapter} label="WebGPU">
                {support.webgpuAdapter
                  ? 'A GPU adapter is available for the in-browser model.'
                  : support.webgpu
                    ? 'The API exists but no adapter was offered; WebAssembly is used.'
                    : 'Not available; WebAssembly is used.'}
              </Check3>
            </ul>
          </Panel>
        </div>
      </div>
    </Page>
  )
}
