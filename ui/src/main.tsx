import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app'
import { Toaster } from './components/ui/sonner'
import { TooltipProvider } from './components/ui/tooltip'
import { ThemeProvider } from './hooks/use-theme'
import './index.css'
import { live } from './lib/live'

async function boot() {
  if (import.meta.env.VITE_BOLO_DEMO === '1') {
    const [{ ensureIsolation }, { installDemo }] = await Promise.all([import('./demo/isolation'), import('./demo')])
    if (await ensureIsolation()) return // reloading to become cross-origin isolated
    installDemo()
  } else if (import.meta.env.VITE_BOLO_MOCK === '1') {
    const { installMock } = await import('./mock')
    installMock()
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
