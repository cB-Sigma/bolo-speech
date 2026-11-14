import { PlacementSelect } from '@/components/placement-select'
import { placeableStt } from '@/lib/browser-stt/models'
import { useVerdict, VerdictLine } from '@/components/model-fit'
import { sttNeeds } from '@/components/stt/language-controls'
import { deleteDownload, downloadedIds } from '@/lib/downloads'
import { Cpu, Download, HardDrive, Trash2, Zap } from 'lucide-react'
import { toast } from 'sonner'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { useStore } from '@/hooks/use-store'
import type { SupportState } from '@/hooks/use-support'
import { sharedEngine } from '@/lib/browser-stt/engine'
import { fmtBytes, fmtMs, languageName } from '@/lib/format'
import { isIndic } from '@/lib/languages'
import { INSTALL_COMMAND } from '@/lib/api/config'
import { LoadProgress, LoadTrouble } from './model-status'

function Fact({ label, ok, children }: { label: string; ok: boolean | null; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="flex items-center gap-1.5 text-right text-foreground">
        <span
          className={`size-1.5 rounded-full ${ok == null ? 'bg-chart-mute' : ok ? 'bg-good' : 'bg-chart-mute'}`}
          aria-hidden
        />
        {children}
      </span>
    </div>
  )
}

/*
 * Measured, not assumed: 13 Hindi recordings (12 from FLEURS dev, 1 synthesised) through the same 8-bit ONNX weights.
 * Figures and transcripts are in docs/measurements/browser-whisper-hindi.json.
 */
const HINDI_MODEL_NOTE =
  'On 13 Hindi test recordings (12 of them real speakers) this model got about 4 words in 10 wrong (word error rate 40%, character error rate 27%). Good enough to see Hindi working, not for real use.'
const MULTILINGUAL_HINDI_NOTE =
  'On the same 13 Hindi recordings Whisper Tiny and Whisper Base got more words wrong than right (word error rate above 100%) and mostly wrote in Latin or Urdu script. Choose Whisper Tiny Hindi instead.'
const MULTILINGUAL_INDIC_NOTE =
  'We have only measured Hindi, where Whisper Tiny and Base were unusable; other Indian languages are unlikely to be better, and the text may come out in the wrong script.'
const WHISPER_GENERAL_NOTE =
  'Whisper Tiny and Base are small general-purpose models: reasonable for major European languages, poor for Indian ones. Audio is decoded a stretch at a time, so text arrives later than with Moonshine.'

function EngineVerdict() {
  const s = useStore(sharedEngine().store)
  const verdict = useVerdict(sttNeeds(s.model, Boolean(s.cached)))
  return <VerdictLine className="mt-2" verdict={verdict} bytes={s.model.approxBytes} />
}

/** Status of the in-browser recogniser: download, cache, and what the browser offers it. */
export function BrowserEnginePanel({
  support,
  language,
  picker,
}: {
  support: SupportState
  /** Language the recogniser is set to (`auto` or a code). */
  language?: string
  /** Model and language controls, when the page wants them in this panel. */
  picker?: React.ReactNode
}) {
  const engine = sharedEngine()
  const s = useStore(engine.store)

  const clear = async () => {
    try {
      for (const id of downloadedIds('stt')) await deleteDownload(id)
      toast.success('Removed the downloaded speech-to-text models from this browser')
    } catch {
      toast.error('Could not clear the cache')
    }
  }

  return (
    <Panel
      title="In-browser model"
      description="Runs on this device with WebGPU or WebAssembly. Audio never leaves the page."
      data-testid="browser-engine"
      actions={
        s.status === 'idle' || s.status === 'error' ? (
          <Button size="sm" variant="outline" onClick={() => void engine.load().catch(() => undefined)}>
            <Download /> {s.cached ? 'Load' : 'Download'}
          </Button>
        ) : s.cached ? (
          <Button size="icon-sm" variant="ghost" aria-label="Remove downloaded models" onClick={() => void clear()}>
            <Trash2 />
          </Button>
        ) : null
      }
    >
      {picker}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate num text-[13px] font-medium">{s.model.label}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            {s.model.multilingual
              ? `${s.model.languages.length} languages`
              : languageName(s.model.languages[0] ?? 'en')}{' '}
            · {fmtBytes(s.model.approxBytes)} · {s.model.license} ·{' '}
            <a
              className="text-link underline underline-offset-2"
              href={s.model.source}
              target="_blank"
              rel="noreferrer"
            >
              source
            </a>
          </div>
        </div>
        <span
          className="shrink-0 rounded-md border border-border px-1.5 py-0.5 text-[11px] font-medium"
          data-testid="engine-status"
          data-status={s.status}
        >
          {s.status === 'ready'
            ? 'Ready'
            : s.status === 'loading'
              ? 'Loading'
              : s.status === 'error'
                ? 'Failed'
                : s.cached
                  ? 'Cached'
                  : 'Not downloaded'}
        </span>
      </div>

      <PlacementSelect
        className="mt-3"
        model={placeableStt(s.model)}
        disabled={s.status === 'loading' || engine.busy}
        onChange={() => {
          // Load again where the visitor asked, if it was loaded.
          if (s.status === 'ready') {
            engine.dispose()
            void engine.load().catch(() => undefined)
          }
        }}
      />
      {s.status !== 'ready' && <EngineVerdict />}
      <LoadProgress className="mt-3" />
      <LoadTrouble className="mt-3" />
      {s.model.family === 'Whisper' && (
        <p
          className="mt-3 rounded-md border border-border bg-muted/40 px-2.5 py-2 text-xs leading-[1.15rem] text-muted-foreground"
          data-testid="whisper-note"
        >
          {!s.model.multilingual ? (
            <>
              <span className="font-medium text-foreground">Expect mistakes.</span> {HINDI_MODEL_NOTE}
            </>
          ) : language === 'hi' ? (
            <>
              <span className="font-medium text-foreground">Not usable for Hindi.</span> {MULTILINGUAL_HINDI_NOTE}
            </>
          ) : language && language !== 'auto' && isIndic(language) ? (
            <>
              <span className="font-medium text-foreground">Expect a rough guess in {languageName(language)}.</span>{' '}
              {MULTILINGUAL_INDIC_NOTE}
            </>
          ) : (
            WHISPER_GENERAL_NOTE
          )}{' '}
          The bolo server runs much better models for Indian languages (larger fine-tuned Hindi Whisper models, the
          IndicConformer community export and others): install it with{' '}
          <span className="num break-all text-foreground">{INSTALL_COMMAND}</span> and pull one from Models.
        </p>
      )}

      <div className="mt-3 divide-y divide-border border-t border-border">
        <Fact label="Runs on" ok={s.status === 'ready' ? true : null}>
          {s.placement === 'split' ? (
            <>
              <Zap className="size-3" aria-hidden /> Encoder on WebGPU, decoder on WebAssembly · {s.threads} thread
              {s.threads === 1 ? '' : 's'}
            </>
          ) : s.device === 'webgpu' ? (
            <>
              <Zap className="size-3" aria-hidden /> WebGPU
            </>
          ) : s.device === 'wasm' ? (
            <>
              <Cpu className="size-3" aria-hidden /> WebAssembly · {s.threads} thread{s.threads === 1 ? '' : 's'}
            </>
          ) : support.webgpuAdapter ? (
            'WebGPU available'
          ) : (
            'WebAssembly'
          )}
        </Fact>
        <Fact label="SIMD" ok={support.wasmSimd}>
          {support.wasmSimd ? 'Available' : 'Unavailable'}
        </Fact>
        <Fact label="Threads" ok={support.wasmThreads}>
          {support.wasmThreads
            ? `Up to ${Math.min(4, support.hardwareConcurrency)}`
            : 'Off: page is not cross-origin isolated'}
        </Fact>
        {s.loadMs != null && (
          <Fact label="Load time" ok>
            <HardDrive className="size-3" aria-hidden /> <span className="num">{fmtMs(s.loadMs)}</span>
          </Fact>
        )}
      </div>
    </Panel>
  )
}
