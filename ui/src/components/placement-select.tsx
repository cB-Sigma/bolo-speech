import { useDevice } from '@/components/model-fit'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useStore } from '@/hooks/use-store'
import {
  PLACEMENT_TEXT,
  placementOptions,
  placements,
  resolvePlacement,
  setPlacementChoice,
  type PlaceableModel,
  type PlacementChoice,
} from '@/lib/placement'
import { cn } from '@/lib/utils'

const RESULT_TEXT = { webgpu: 'WebGPU', wasm: 'CPU', split: 'split' } as const

/**
 * Where one in-page model runs: Auto, all WebGPU, all CPU, or split (encoder
 * or front half on WebGPU, the rest on the CPU). A choice that cannot work
 * here is warned about, not refused. `onChange` reloads the model if needed.
 */
export function PlacementSelect({
  model,
  onChange,
  disabled,
  className,
}: {
  model: PlaceableModel
  onChange?: (choice: PlacementChoice) => void
  disabled?: boolean
  className?: string
}) {
  const d = useDevice()
  const saved = useStore(placements)
  const choice = saved.choices[model.id] ?? 'auto'
  const options = placementOptions(model)
  const auto = resolvePlacement(model, 'auto', d)
  const gpu = d.webgpu
  const shared = d.platform !== 'desktop' || /apple/i.test(gpu?.adapter ?? '')
  const warning =
    gpu && !gpu.available && (choice === 'webgpu' || choice === 'split')
      ? 'No WebGPU in this browser: it will run on the CPU.'
      : model.requiresWebGpu && (choice === 'wasm' || choice === 'split')
        ? 'This model has 16-bit weights for WebGPU: on the CPU it is likely to fail.'
        : null
  const label = (c: PlacementChoice) =>
    c === 'auto'
      ? `Auto (${RESULT_TEXT[auto.placement]})`
      : c === 'split' && model.splitText
        ? `Split: ${model.splitText}`
        : PLACEMENT_TEXT[c]
  return (
    <div className={cn('grid gap-1 text-xs', className)} data-testid="placement">
      <label className="flex min-w-0 flex-wrap items-center gap-2">
        <span className="text-muted-foreground">Run on</span>
        <Select
          value={options.includes(choice) ? choice : 'auto'}
          onValueChange={(v) => {
            setPlacementChoice(model.id, v as PlacementChoice)
            onChange?.(v as PlacementChoice)
          }}
          disabled={disabled}
        >
          <SelectTrigger
            className="h-7 w-auto max-w-full min-w-28"
            aria-label={`Where ${model.id} runs`}
            data-testid="placement-select"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" align="start" className="w-[min(22rem,calc(100vw-2rem))]">
            {options.map((c) => (
              <SelectItem key={c} value={c} data-testid={`placement-${c}`}>
                {label(c)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      {choice === 'auto' && auto.why && (
        <p className="text-muted-foreground" data-testid="placement-why">
          Auto: {RESULT_TEXT[auto.placement]}, {auto.why}.
        </p>
      )}
      {warning && (
        <p className="text-warning-text" data-testid="placement-warning">
          {warning}
        </p>
      )}
      {(choice === 'split' || auto.placement === 'split') && shared && gpu?.available && (
        <p className="text-muted-foreground" data-testid="placement-shared">
          This device&apos;s GPU shares memory with everything else: a split helps with WebGPU&apos;s limits and with
          speed, not with total memory.
        </p>
      )}
    </div>
  )
}
