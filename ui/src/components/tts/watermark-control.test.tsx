import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import type { CatalogModel } from '@/lib/api/types'
import { WatermarkControl, watermarkParam, type WatermarkChoice } from './watermark-control'

const model = (id: string, license: string) =>
  ({
    id,
    license,
    task: 'watermark',
    engine: 'onnxruntime',
    requires: [],
    installed: false,
    loaded: false,
    size_bytes: 90e6,
    features: [],
  }) as unknown as CatalogModel

function Harness({ models }: { models: CatalogModel[] }) {
  const [value, setValue] = useState<WatermarkChoice>(null)
  return (
    <>
      <WatermarkControl value={value} onChange={setValue} models={models} defaultModel="audioseal-16bits" />
      <output data-testid="sent">{String(watermarkParam(value))}</output>
    </>
  )
}

describe('watermark control', () => {
  it('sends nothing when off, on for the default, the model id when one is picked', () => {
    expect(watermarkParam(null)).toBeUndefined()
    expect(watermarkParam('')).toBe('on')
    expect(watermarkParam('perth-onnx')).toBe('perth-onnx')
  })

  it('is off by default and offers the default and every catalog watermarker once switched on', async () => {
    const user = userEvent.setup()
    render(<Harness models={[model('audioseal-16bits', 'MIT'), model('wavmark-onnx', 'MIT')]} />)
    expect(screen.getByTestId('sent')).toHaveTextContent('undefined')
    expect(screen.queryByTestId('watermark-model')).toBeNull()
    await user.click(screen.getByTestId('watermark'))
    expect(screen.getByTestId('sent')).toHaveTextContent('on')
    expect(screen.getByTestId('watermark-model')).toHaveTextContent('Default (audioseal-16bits)')
    await user.click(screen.getByTestId('watermark-model'))
    expect(await screen.findByTestId('watermark-model-wavmark-onnx')).toHaveTextContent('MIT')
    await user.click(screen.getByTestId('watermark-model-wavmark-onnx'))
    expect(screen.getByTestId('sent')).toHaveTextContent('wavmark-onnx')
  })
})
