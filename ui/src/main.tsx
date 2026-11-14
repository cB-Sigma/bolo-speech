import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app'
import { Toaster } from './components/ui/sonner'
import { TooltipProvider } from './components/ui/tooltip'
import { ThemeProvider } from './hooks/use-theme'
import './index.css'
import { installErrorCapture } from './lib/diagnostics'
import { live } from './lib/live'

async function installFakeStt() {
  const { FakeSpeechWorker, readFakeSttOptions } = await import('./mock/fake-stt-worker')
  const fake = readFakeSttOptions()
  if (fake) {
    const { setSpeechWorkerFactory } = await import('./lib/browser-stt/engine')
    setSpeechWorkerFactory(() => new FakeSpeechWorker(fake))
  }
}

async function boot() {
  installErrorCapture()
  if (import.meta.env.VITE_BOLO_DEMO === '1') {
    const [{ ensureIsolation }, { installDemo }] = await Promise.all([import('./demo/isolation'), import('./demo')])
    if (await ensureIsolation()) return // reloading to become cross-origin isolated
    installDemo()
    // Tests only: a scripted recogniser instead of a real model, when the page is told to use one.
    let fake = false
    try {
      fake = Boolean(localStorage.getItem('bolo.mock.browserStt'))
    } catch {
      // storage blocked: the real recogniser
    }
    if (fake) await installFakeStt()
    try {
      if (localStorage.getItem('bolo.mock.diarize')) {
        const [{ FakeDiarizeWorker }, { setDiarizeWorkerFactory }] = await Promise.all([
          import('./mock/fake-diarize-worker'),
          import('./lib/browser-diarize/engine'),
        ])
        setDiarizeWorkerFactory(() => new FakeDiarizeWorker())
      }
      if (localStorage.getItem('bolo.mock.llm')) {
        const [{ FakeLlmWorker }, { setLlmWorkerFactory }] = await Promise.all([
          import('./mock/fake-llm-worker'),
          import('./lib/browser-llm/engine'),
        ])
        setLlmWorkerFactory(() => new FakeLlmWorker())
      }
    } catch {
      // storage blocked: the real diarizer
    }
  } else if (import.meta.env.VITE_BOLO_MOCK === '1') {
    const { installMock } = await import('./mock')
    installMock()
    await installFakeStt()
  }
  const queryClient = new QueryClient({
    defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
  })
  live.start(queryClient)
  createRoot(document.getElementById('root') as HTMLElement).render(
    <StrictMode>
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <TooltipProvider delayDuration={250}>
            <App />
            <Toaster position="bottom-right" />
          </TooltipProvider>
        </QueryClientProvider>
      </ThemeProvider>
    </StrictMode>,
  )
}

void boot()
